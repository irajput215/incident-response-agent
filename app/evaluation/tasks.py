"""Loading the evaluation suite — and refusing to run a broken one.

The reference implementation silently falls back to a built-in task list when its
suite file is missing or malformed. For a CI gate that is the worst possible
behaviour: a typo means your new task never runs, while the suite reports green.
**A failure that looks like success is worse than a failure.**

So this loader raises. Every malformed input is a hard error, including the ones
that would be merely annoying in other contexts: an unknown ``expect`` key (a
typo'd assertion that silently does nothing), a duplicate task id, a task that
names neither a scenario nor an inline incident.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from app.scenarios import SCENARIOS_BY_ID

DEFAULT_TASKS_PATH = Path("evals/tasks.yaml")

#: Keys that control the harness rather than asserting something about a run.
#: `min_checks` enforces the "a task with an empty expect block must fail" rule.
CONTROL_KEYS: frozenset[str] = frozenset({"min_checks"})


class SuiteConfigError(ValueError):
    """The suite file is missing, malformed, or asserts something unrecognised."""


@dataclass(frozen=True, slots=True)
class EvalTask:
    """One evaluation case: an incident, and what a correct response looks like."""

    id: str
    description: str
    incident: dict[str, Any]
    expect: dict[str, Any]
    approve: bool = True
    defaults: dict[str, Any] = field(default_factory=dict)

    def expected(self, key: str) -> Any:
        """The task's own expectation, falling back to the suite defaults."""
        if key in self.expect:
            return self.expect[key]
        return self.defaults.get(key)

    def has(self, key: str) -> bool:
        return key in self.expect or key in self.defaults


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SuiteConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def load_suite(path: Path | str | None = None) -> tuple[list[EvalTask], dict[str, Any]]:
    """Load and fully validate the suite. Raises :class:`SuiteConfigError`.

    Returns ``(tasks, defaults)``. Validation happens here rather than at
    evaluation time so a broken suite fails in the first second of CI, not after
    a ten-minute agent run.
    """
    from app.evaluation.evaluators import EVALUATOR_KEYS

    resolved = Path(path) if path is not None else DEFAULT_TASKS_PATH
    if not resolved.exists():
        raise SuiteConfigError(
            f"evaluation suite not found at {resolved}. "
            "Refusing to substitute a built-in suite: a gate that silently runs "
            "different tasks than you wrote is not a gate."
        )

    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SuiteConfigError(f"{resolved} is not valid YAML: {exc}") from exc

    document = _require_mapping(raw, str(resolved))

    unknown_top_level = set(document) - {"version", "defaults", "tasks"}
    if unknown_top_level:
        raise SuiteConfigError(
            f"unknown top-level key(s) in {resolved}: {sorted(unknown_top_level)}"
        )

    defaults = _require_mapping(document.get("defaults") or {}, "defaults")
    _validate_expect_keys(defaults, "defaults", EVALUATOR_KEYS)

    raw_tasks = document.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise SuiteConfigError(f"{resolved} defines no tasks")

    tasks: list[EvalTask] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw_tasks):
        where = f"tasks[{index}]"
        entry = _require_mapping(entry, where)

        task_id = entry.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise SuiteConfigError(f"{where} has no id")
        if task_id in seen:
            raise SuiteConfigError(f"duplicate task id {task_id!r}")
        seen.add(task_id)

        unknown = set(entry) - {
            "id",
            "description",
            "scenario",
            "incident",
            "incident_overrides",
            "expect",
            "approve",
        }
        if unknown:
            raise SuiteConfigError(f"{where} ({task_id}) has unknown key(s): {sorted(unknown)}")

        incident = _resolve_incident(entry, where)
        expect = _require_mapping(entry.get("expect") or {}, f"{where}.expect")
        _validate_expect_keys(expect, f"{where}.expect", EVALUATOR_KEYS)

        tasks.append(
            EvalTask(
                id=task_id,
                description=str(entry.get("description") or "").strip(),
                incident=incident,
                expect=expect,
                approve=bool(entry.get("approve", True)),
                defaults=defaults,
            )
        )

    return tasks, defaults


def _resolve_incident(entry: dict[str, Any], where: str) -> dict[str, Any]:
    """Build the alert from a scenario reference, an inline payload, or both."""
    scenario_id = entry.get("scenario")
    inline = entry.get("incident")
    overrides = entry.get("incident_overrides") or {}

    if scenario_id is None and inline is None:
        raise SuiteConfigError(f"{where} names neither `scenario` nor `incident`")

    if scenario_id is not None:
        if scenario_id not in SCENARIOS_BY_ID:
            raise SuiteConfigError(
                f"{where} references unknown scenario {scenario_id!r}; "
                f"known: {sorted(SCENARIOS_BY_ID)}"
            )
        incident = dict(SCENARIOS_BY_ID[scenario_id].incident)
        incident.update(_require_mapping(overrides, f"{where}.incident_overrides"))
        if inline is not None:
            incident.update(_require_mapping(inline, f"{where}.incident"))
        return incident

    incident = dict(_require_mapping(inline, f"{where}.incident"))
    incident.update(_require_mapping(overrides, f"{where}.incident_overrides"))
    return incident


def _validate_expect_keys(expect: dict[str, Any], where: str, known: frozenset[str]) -> None:
    unknown = set(expect) - known - CONTROL_KEYS
    if unknown:
        raise SuiteConfigError(
            f"{where} asserts unknown key(s): {sorted(unknown)}. "
            f"Known assertions: {sorted(known)}; control keys: {sorted(CONTROL_KEYS)}. "
            "An unrecognised assertion would silently never run."
        )


def with_tasks(tasks: list[EvalTask], ids: list[str]) -> list[EvalTask]:
    """Select a subset by id. An unknown id is an error, not an empty selection."""
    by_id = {task.id: task for task in tasks}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SuiteConfigError(f"unknown task id(s): {missing}; known: {sorted(by_id)}")
    return [replace(by_id[i]) for i in ids]


__all__ = [
    "DEFAULT_TASKS_PATH",
    "EvalTask",
    "SuiteConfigError",
    "load_suite",
    "with_tasks",
]
