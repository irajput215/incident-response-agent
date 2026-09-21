"""The evaluation harness: run the suite, report with real numbers, gate CI.

This is the highest-value part of the project. Most portfolio LLM systems have no
evaluation at all, which means every change is a guess; here, a change that makes
the agent worse fails a build.

Three properties are deliberate:

**It runs with no API key.** The offline analyst is the default, so the gate is
meaningful on a clean machine and in a fork's CI. A gate that only passes on the
author's laptop is not a gate.

**The report carries actual numbers.** Every check renders as ``"5 >= 5"`` or
``"UPSTREAM_DEPENDENCY_FAILURE == SCHEMA_CHANGE"``, so a red build tells you what
happened rather than that something did.

**A task with nothing to assert fails.** ``all([])`` is ``True`` in Python, so a
task whose ``expect`` block was emptied — by a bad merge, by accident — would
otherwise sail through and quietly weaken the suite. It is counted and rejected
explicitly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import psycopg
from langgraph.checkpoint.memory import MemorySaver
from psycopg.rows import dict_row

from app.agents import IncidentAgent
from app.config import Settings, get_settings
from app.db import IncidentRepository, ensure_schema
from app.evaluation.evaluators import Check, Observation, run_evaluator
from app.evaluation.tasks import (
    DEFAULT_TASKS_PATH,
    EvalTask,
    SuiteConfigError,
    load_suite,
    with_tasks,
)
from app.llm import LLMClient, build_llm
from app.observability import Metrics, get_logger, log, setup_logging
from app.schemas import ApprovalDecision, IncidentCreate
from app.tools import build_context, build_registry

_log = get_logger("app.evaluation")

#: Default gate: **every** task must pass. Lowering it tolerates failures, which
#: is occasionally right for a noisy model-backed suite and never right for the
#: deterministic one, so the default is the strict reading.
DEFAULT_MIN_ACCURACY = 1.0


@dataclass
class TaskResult:
    """One evaluated task."""

    task_id: str
    description: str
    passed: bool
    checks: list[Check]
    duration_ms: float = 0.0
    status: str = ""
    error: str | None = None
    observation: Observation | None = None

    @property
    def failed_checks(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]


@dataclass
class SuiteReport:
    """The result of a whole suite run."""

    results: list[TaskResult] = field(default_factory=list)
    model: str = ""
    tasks_path: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    finished_at: datetime | None = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def accuracy(self) -> float:
        """Fraction of tasks that passed. Zero for an empty suite, not 1.0."""
        return self.passed_count / self.total if self.total else 0.0

    @property
    def passed(self) -> bool:
        return self.total > 0 and self.passed_count == self.total

    @property
    def total_llm_calls(self) -> int:
        return sum(r.observation.llm_calls for r in self.results if r.observation)

    @property
    def total_tokens(self) -> int:
        return sum(r.observation.total_tokens for r in self.results if r.observation)

    @property
    def total_duration_ms(self) -> float:
        return sum(r.duration_ms for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tasks_path": self.tasks_path,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "total": self.total,
            "passed": self.passed_count,
            "accuracy": round(self.accuracy, 4),
            "suite_passed": self.passed,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens": self.total_tokens,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "results": [
                {
                    "task_id": r.task_id,
                    "passed": r.passed,
                    "status": r.status,
                    "error": r.error,
                    "duration_ms": round(r.duration_ms, 1),
                    "checks": [
                        {"name": c.name, "ok": c.ok, "detail": c.detail} for c in r.checks
                    ],
                }
                for r in self.results
            ],
        }

    def render(self) -> str:
        """A human-readable report where every check shows its actual numbers."""
        lines: list[str] = []
        lines.append("")
        lines.append("=" * 78)
        lines.append(f"AGENT EVALUATION  ·  model={self.model}  ·  suite={self.tasks_path}")
        lines.append("=" * 78)

        for result in self.results:
            mark = "PASS" if result.passed else "FAIL"
            lines.append("")
            lines.append(f"[{mark}] {result.task_id}  ({result.duration_ms:.0f}ms)  {result.status}")
            if result.description:
                for chunk in _wrap(result.description, 72):
                    lines.append(f"       {chunk}")
            for check in result.checks:
                flag = " ok " if check.ok else "FAIL"
                lines.append(f"       {flag}  {check.name:<24} {check.detail}")
            if result.error:
                lines.append(f"       error: {result.error}")

        lines.append("")
        lines.append("-" * 78)
        lines.append(
            f"accuracy {self.passed_count}/{self.total} = {self.accuracy:.0%}"
            f"   ·   llm calls {self.total_llm_calls}"
            f"   ·   tokens {self.total_tokens}"
            f"   ·   {self.total_duration_ms / 1000:.1f}s"
        )
        lines.append("SUITE PASSED" if self.passed else "SUITE FAILED")
        lines.append("-" * 78)
        return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


class EvaluationSuite:
    """Runs the evaluation tasks against a real agent on a real database."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        tasks_path: Path | str | None = None,
        llm: LLMClient | None = None,
        connection: psycopg.Connection[dict[str, Any]] | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.tasks_path = Path(tasks_path) if tasks_path is not None else DEFAULT_TASKS_PATH
        self.tasks, self.defaults = load_suite(self.tasks_path)  # raises on a broken suite
        self.llm = llm
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()
        self._owns_connection = connection is None
        self._conn = connection or psycopg.connect(
            self.settings.database_url,
            row_factory=dict_row,
            connect_timeout=5,
        )
        ensure_schema(self._conn)
        self.repo = IncidentRepository(self._conn)

    # --- running -----------------------------------------------------------
    def run(self, task_ids: list[str] | None = None) -> SuiteReport:
        tasks = with_tasks(self.tasks, task_ids) if task_ids else list(self.tasks)
        report = SuiteReport(model="", tasks_path=str(self.tasks_path))

        for task in tasks:
            result = self._run_task(task)
            if not report.model and result.observation is not None:
                report.model = self.llm.model if self.llm is not None else "stub"
            report.results.append(result)

        if not report.model:
            report.model = self.llm.model if self.llm is not None else "stub"
        report.finished_at = datetime.now(tz=UTC)
        log(
            _log,
            20,
            "evaluation_finished",
            passed=report.passed_count,
            total=report.total,
            accuracy=round(report.accuracy, 4),
        )
        return report

    def _run_task(self, task: EvalTask) -> TaskResult:
        """Run one task with completely fresh state.

        Per-task metrics and a truncated platform schema. Without this, a task
        could pass because the previous task left the right rows behind — the
        subtlest way for an evaluation suite to start lying.
        """
        metrics = Metrics()
        self.repo.reset()

        llm = self.llm or build_llm(self.settings, metrics=metrics)
        registry = build_registry(build_context(self.settings, metrics=metrics))
        agent = IncidentAgent(
            settings=self.settings,
            llm=llm,
            registry=registry,
            repository=self.repo,
            metrics=metrics,
            checkpointer=self.checkpointer,
        )

        started = perf_counter()
        error: str | None = None
        outcome = None
        # Whether the graph *paused* is a different question from whether it is
        # paused now: after a resume the final state is running again. Record the
        # pause when it happens, because "did this action require a human?" is
        # the behavioural property being evaluated.
        paused = False
        try:
            alert = IncidentCreate.model_validate(task.incident)
            outcome = agent.investigate(alert)

            if outcome.awaiting_approval:
                paused = True
                decision = ApprovalDecision(
                    approved=task.approve,
                    approver="eval-harness",
                    note="evaluation run",
                )
                outcome = agent.resume(outcome.investigation_id, decision)
        except Exception as exc:  # a crashing task is a failed task, not a crashed suite
            error = f"{type(exc).__name__}: {exc}"
            log(_log, 40, "evaluation_task_crashed", task=task.id, error=error[:200])
        duration_ms = (perf_counter() - started) * 1000

        observation = self._observe(task, outcome, metrics, duration_ms, error, paused=paused)
        checks = self._check(task, observation)
        passed = bool(checks) and all(c.ok for c in checks)

        log(
            _log,
            20,
            "evaluation_task_finished",
            task=task.id,
            passed=passed,
            failed=[c.name for c in checks if not c.ok],
        )
        return TaskResult(
            task_id=task.id,
            description=task.description,
            passed=passed,
            checks=checks,
            duration_ms=duration_ms,
            status=observation.status,
            error=error,
            observation=observation,
        )

    def _observe(
        self,
        task: EvalTask,
        outcome: Any,
        metrics: Metrics,
        duration_ms: float,
        error: str | None,
        *,
        paused: bool = False,
    ) -> Observation:
        observation = Observation(task_id=task.id, duration_ms=duration_ms, error=error)
        observation.interrupted = paused
        if outcome is None:
            return observation

        observation.status = outcome.status.value

        counters = metrics.snapshot()["counters"]
        observation.write_attempts = int(counters.get("tool.run_sql.rejected", 0))
        observation.unknown_tool_calls = int(counters.get("tool.unknown", 0))

        # Tool selection is read from the recorded calls, not from metrics: the
        # evaluation cares *which* tools ran, and a counter cannot say.
        calls = self.repo.get_tool_calls(outcome.investigation_id)
        observation.tool_names = [str(call["tool"]) for call in calls]

        report = outcome.report
        if report is not None:
            observation.category = report.category.value if report.category else None
            if report.remediation:
                observation.action = report.remediation.action.value
                observation.target = report.remediation.target
            if report.root_cause:
                observation.confidence = report.root_cause.confidence
                observation.conclusive = report.root_cause.is_conclusive
            observation.evidence_count = len(report.evidence)
            observation.llm_calls = report.llm_calls
            observation.tool_calls = report.tool_calls
            observation.tokens_in = report.tokens_in
            observation.tokens_out = report.tokens_out
            observation.report_summary = report.summary
        else:
            # No report: still record what the metrics know, so a failing task's
            # checks report real numbers instead of zeros.
            observation.tokens_in = int(counters.get("llm.tokens_in", 0))
            observation.tokens_out = int(counters.get("llm.tokens_out", 0))
            observation.llm_calls = int(counters.get("llm.calls", 0))
            observation.tool_calls = len(calls)

        return observation

    def _check(self, task: EvalTask, observation: Observation) -> list[Check]:
        """Run every assertion, and refuse to pass a task that asserts nothing."""
        checks: list[Check] = []
        authored = 0

        for key, expected in task.expect.items():
            checks.append(run_evaluator(key, observation, expected))
            authored += 1

        # Suite-wide safety properties, unless the task overrides them.
        for key, expected in task.defaults.items():
            if key in task.expect or key == "min_checks":
                continue
            checks.append(run_evaluator(key, observation, expected))

        minimum = int(task.expected("min_checks") or 1)
        if authored < minimum:
            checks.append(
                Check(
                    "has_expectations",
                    False,
                    f"{authored} task assertion(s) < {minimum} required — an empty "
                    "expect block must fail, not pass vacuously",
                )
            )
        return checks

    def close(self) -> None:
        if self._owns_connection:
            self._conn.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation.harness",
        description="Run the agent evaluation suite and gate on accuracy.",
    )
    parser.add_argument("--suite", default=str(DEFAULT_TASKS_PATH), help="Path to tasks.yaml")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=DEFAULT_MIN_ACCURACY,
        help=f"Fail below this fraction of tasks (default {DEFAULT_MIN_ACCURACY})",
    )
    parser.add_argument("--task", action="append", dest="task_ids", help="Run only this task id")
    parser.add_argument("--report", help="Write the JSON report to this path")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, json_output=settings.log_json)

    try:
        suite = EvaluationSuite(settings, tasks_path=args.suite)
    except SuiteConfigError as exc:
        # Loud, and before any agent runs: a broken suite is a configuration
        # failure, and pretending otherwise is how a gate rots.
        print(f"EVALUATION SUITE INVALID: {exc}", file=sys.stderr)
        return 2

    try:
        report = suite.run(args.task_ids)
    finally:
        suite.close()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())

    if args.report:
        Path(args.report).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nreport written to {args.report}")

    if report.accuracy < args.min_accuracy:
        print(
            f"\nFAIL: accuracy {report.accuracy:.0%} is below the required "
            f"{args.min_accuracy:.0%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
