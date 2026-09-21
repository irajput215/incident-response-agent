"""Evaluation: the suite that gates CI, and the tracing that explains it.

Two halves, deliberately separable:

* :mod:`app.evaluation.harness` runs the task suite locally and returns a report
  with real numbers. It needs no API key and no network, which is what makes the
  CI gate meaningful.
* :mod:`app.tracing` wires LangSmith. It lives at the top level rather than here
  because every layer uses it, and nesting it under evaluation created a real
  import cycle. With no key configured it is a no-op, so the same code path is
  exercised everywhere and the tracing configuration cannot rot between the
  machines that have a key and the ones that do not.
"""

from app.evaluation.evaluators import EVALUATOR_KEYS, EVALUATORS, Check, Observation
from app.evaluation.harness import EvaluationSuite, SuiteReport, TaskResult
from app.evaluation.tasks import EvalTask, SuiteConfigError, load_suite

__all__ = [
    "EVALUATORS",
    "EVALUATOR_KEYS",
    "Check",
    "EvalTask",
    "EvaluationSuite",
    "Observation",
    "SuiteConfigError",
    "SuiteReport",
    "TaskResult",
    "load_suite",
]
