"""The offline analyst: a baseline that must be measurably right, and honest.

Two kinds of test here.

**Classification.** The baseline is scored against a table of realistic failure
messages. If this drops, either the analyst regressed or the fixture changed —
and because the same fixtures drive the agent's evaluation, a failure here
explains a drop there.

**Refusal.** The most important behaviour is that the analyst says
``is_conclusive=False`` when the evidence does not support an answer. A baseline
that always guesses would score better on a naive accuracy metric and be worse
at everything that matters.
"""
from __future__ import annotations

import json

import pytest

from app.llm import HeuristicLLM
from app.llm.heuristic import detect_category, detect_severity
from app.schemas import FailureCategory, Remediation, RemediationAction, RootCause, Severity, Triage

CLASSIFIER_CASES = [
    ("SparkException: File not found: s3://lake/claims/dt=2026-09-20/ (NoSuchKey)", "DATA_SOURCE_FAILURE"),
    ("AnalysisException: cannot resolve 'policy_type' given input columns: [policy_id, policy_category]", "SCHEMA_CHANGE"),
    ("java.lang.OutOfMemoryError: Java heap space", "RESOURCE_EXHAUSTION"),
    ("Container killed by YARN for exceeding memory limits", "RESOURCE_EXHAUSTION"),
    ("psycopg.OperationalError: connection to server failed: Connection refused", "TRANSIENT_INFRASTRUCTURE"),
    ("HTTPError: 503 Server Error: Service Unavailable", "TRANSIENT_INFRASTRUCTURE"),
    ("AccessDenied: not authorized to perform s3:GetObject", "CONFIGURATION_ERROR"),
    ("TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'", "CODE_ERROR"),
    ("dbt test failure: null rate 0.31 exceeds threshold 0.05", "DATA_QUALITY_FAILURE"),
    ("upstream ingestion run did not complete for dt=2026-09-20", "UPSTREAM_DEPENDENCY_FAILURE"),
    ("the parent task failed, so this task was skipped", "UPSTREAM_DEPENDENCY_FAILURE"),
    ("something entirely unexpected happened", "UNKNOWN"),
]

REMEDIATION_CASES = [
    ("DATA_SOURCE_FAILURE", RemediationAction.BACKFILL_PARTITION),
    ("UPSTREAM_DEPENDENCY_FAILURE", RemediationAction.RERUN_UPSTREAM),
    ("TRANSIENT_INFRASTRUCTURE", RemediationAction.RERUN_PIPELINE),
    ("RESOURCE_EXHAUSTION", RemediationAction.RERUN_PIPELINE),
    ("SCHEMA_CHANGE", RemediationAction.ALERT_OWNER),
    ("CONFIGURATION_ERROR", RemediationAction.ALERT_OWNER),
    ("DATA_QUALITY_FAILURE", RemediationAction.ESCALATE),
    ("CODE_ERROR", RemediationAction.ESCALATE),
    ("UNKNOWN", RemediationAction.ESCALATE),
]


def incident(**overrides: object) -> dict:
    payload = {
        "pipeline": "customer_claims_daily",
        "run_id": "run_98321",
        "status": "FAILED",
        "error": "SparkException: File not found: s3://acme-lake/claims-raw/dt=2026-09-20/",
        "table": "claims",
    }
    payload.update(overrides)
    return payload


def messages_for(payload: dict) -> list[dict]:
    return [{"role": "user", "content": f"INCIDENT:\n{json.dumps(payload)}"}]


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", CLASSIFIER_CASES)
def test_failure_signatures_are_classified(text: str, expected: str) -> None:
    assert detect_category(text).value == expected


def test_upstream_pattern_tolerates_json_punctuation() -> None:
    """The evidence arrives as JSON, so the gap between "upstream" and the verb
    has to survive quotes, colons and commas."""
    assert (
        detect_category('"upstream_pipeline": "claims_ingestion", "status": "FAILED"')
        is FailureCategory.UPSTREAM_DEPENDENCY_FAILURE
    )
    assert (
        detect_category('upstream task "claims_ingestion" failed')
        is FailureCategory.UPSTREAM_DEPENDENCY_FAILURE
    )


def test_upstream_pattern_does_not_cross_brackets_or_lines() -> None:
    """The documented limit of the text classifier.

    A gap containing ``[`` or ``]`` stops the match, and the pattern never spans
    a newline — both deliberate, because a loose gap would link an unrelated
    "failed" to the word "upstream" and blame the wrong team. The JSON-shaped
    case that this cannot express is handled structurally instead, by joining
    the dependency list with the upstream run history; see
    ``test_root_cause_joins_upstream_metadata_with_upstream_history``.
    """
    assert (
        detect_category('"upstream": ["claims_ingestion"], "note": "failed"')
        is FailureCategory.UNKNOWN
    )
    assert (
        detect_category("upstream is fine\nsomething unrelated failed")
        is FailureCategory.UNKNOWN
    )


def test_severity_escalates_on_data_loss_wording() -> None:
    assert detect_severity("possible data loss in the claims table", FailureCategory.UNKNOWN) is Severity.CRITICAL


# --------------------------------------------------------------------------- #
# triage
# --------------------------------------------------------------------------- #
def test_triage_matches_the_map_s_example(llm: HeuristicLLM) -> None:
    triage = llm.structured(messages_for(incident()), Triage)
    assert isinstance(triage, Triage)
    assert triage.category is FailureCategory.DATA_SOURCE_FAILURE
    assert triage.severity is Severity.HIGH
    assert triage.initial_hypothesis


def test_triage_admits_uncertainty_on_an_unrecognised_error(llm: HeuristicLLM) -> None:
    triage = llm.structured(messages_for(incident(error="xyzzy plugh")), Triage)
    assert triage.category is FailureCategory.UNKNOWN
    assert triage.confidence < 0.3


# --------------------------------------------------------------------------- #
# root cause
# --------------------------------------------------------------------------- #
def test_root_cause_is_inconclusive_without_evidence(llm: HeuristicLLM) -> None:
    """The refusal is the feature: it drives the graph's next round."""
    root_cause = llm.structured(messages_for(incident(error="xyzzy plugh")), RootCause)
    assert root_cause.is_conclusive is False
    assert root_cause.category is FailureCategory.UNKNOWN
    assert root_cause.confidence < 0.3


def test_root_cause_uses_the_jobs_own_error_when_it_names_a_mechanism(llm: HeuristicLLM) -> None:
    """Direct evidence beats circumstantial: an OOM message is not an upstream problem."""
    messages = messages_for(incident(error="java.lang.OutOfMemoryError: Java heap space"))
    messages.append(
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "ok": True,
                    "tool": "get_pipeline_metadata",
                    "data": {"pipeline": "customer_claims_daily", "upstream": ["claims_ingestion"]},
                }
            ),
        }
    )
    root_cause = llm.structured(messages, RootCause)
    assert root_cause.category is FailureCategory.RESOURCE_EXHAUSTION
    assert root_cause.is_conclusive is True


def test_root_cause_joins_upstream_metadata_with_upstream_history(llm: HeuristicLLM) -> None:
    """Neither tool result alone is enough — the conclusion needs both joined."""
    messages = messages_for(incident())
    messages.append(
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "ok": True,
                    "tool": "get_pipeline_metadata",
                    "data": {"pipeline": "customer_claims_daily", "upstream": ["claims_ingestion"]},
                }
            ),
        }
    )
    messages.append(
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "ok": True,
                    "tool": "get_previous_runs",
                    "data": {
                        "pipeline": "claims_ingestion",
                        "runs": [{"run_id": "run_98050", "status": "FAILED"}],
                    },
                }
            ),
        }
    )
    root_cause = llm.structured(messages, RootCause)
    assert root_cause.category is FailureCategory.UPSTREAM_DEPENDENCY_FAILURE
    assert root_cause.is_conclusive is True
    assert root_cause.confidence >= 0.85


def test_upstream_conclusion_needs_an_actual_failure(llm: HeuristicLLM) -> None:
    """A healthy upstream must not be blamed just for existing."""
    messages = messages_for(incident())
    messages.append(
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "ok": True,
                    "tool": "get_pipeline_metadata",
                    "data": {"pipeline": "customer_claims_daily", "upstream": ["claims_ingestion"]},
                }
            ),
        }
    )
    messages.append(
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "ok": True,
                    "tool": "get_previous_runs",
                    "data": {
                        "pipeline": "claims_ingestion",
                        "runs": [{"run_id": "run_1", "status": "SUCCESS"}],
                    },
                }
            ),
        }
    )
    root_cause = llm.structured(messages, RootCause)
    assert root_cause.category is FailureCategory.DATA_SOURCE_FAILURE


# --------------------------------------------------------------------------- #
# remediation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("category,expected", REMEDIATION_CASES)
def test_remediation_follows_the_established_root_cause(
    llm: HeuristicLLM, category: str, expected: RemediationAction
) -> None:
    """Re-classifying the whole prompt was a real bug: it reported "upstream
    dependency" for every incident, because every pipeline's metadata mentions an
    upstream. The root cause is stated in the prompt; it must be read from there."""
    messages = [
        {"role": "user", "content": f"INCIDENT:\n{json.dumps(incident())}"},
        {
            "role": "user",
            "content": f"ROOT CAUSE:\n{json.dumps({'root_cause': 'x', 'category': category})}",
        },
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "ok": True,
                    "tool": "get_pipeline_metadata",
                    "data": {"upstream": ["some_other_pipeline"]},
                }
            ),
        },
    ]
    remediation = llm.structured(messages, Remediation)
    assert remediation.action is expected


def test_rerun_upstream_targets_the_upstream_not_the_failed_job(llm: HeuristicLLM) -> None:
    """Rerunning the failed job would fail again — its input is still missing."""
    messages = [
        {"role": "user", "content": f"INCIDENT:\n{json.dumps(incident())}"},
        {
            "role": "user",
            "content": (
                "ROOT CAUSE:\n"
                + json.dumps(
                    {"root_cause": "upstream failed", "category": "UPSTREAM_DEPENDENCY_FAILURE"}
                )
            ),
        },
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "ok": True,
                    "tool": "get_pipeline_metadata",
                    "data": {"upstream": ["claims_ingestion"]},
                }
            ),
        },
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "ok": True,
                    "tool": "get_previous_runs",
                    "data": {
                        "pipeline": "claims_ingestion",
                        "runs": [{"run_id": "r", "status": "FAILED"}],
                    },
                }
            ),
        },
    ]
    remediation = llm.structured(messages, Remediation)
    assert remediation.action is RemediationAction.RERUN_UPSTREAM
    assert remediation.target == "claims_ingestion"


def test_data_changing_actions_always_require_approval(llm: HeuristicLLM) -> None:
    """The safe default: anything that mutates data asks a human first."""
    for category in ("UPSTREAM_DEPENDENCY_FAILURE", "DATA_SOURCE_FAILURE", "TRANSIENT_INFRASTRUCTURE"):
        messages = [
            {"role": "user", "content": f"INCIDENT:\n{json.dumps(incident())}"},
            {"role": "user", "content": f"ROOT CAUSE:\n{json.dumps({'category': category})}"},
        ]
        assert llm.structured(messages, Remediation).requires_approval is True


def test_backfill_recovers_the_partition_from_the_evidence(llm: HeuristicLLM) -> None:
    messages = [
        {"role": "user", "content": f"INCIDENT:\n{json.dumps(incident())}"},
        {
            "role": "user",
            "content": f"ROOT CAUSE:\n{json.dumps({'category': 'DATA_SOURCE_FAILURE'})}",
        },
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "ok": True,
                    "tool": "check_row_count",
                    "data": {"expected_partition": "2026-09-20", "row_count": 0},
                }
            ),
        },
    ]
    remediation = llm.structured(messages, Remediation)
    assert remediation.action is RemediationAction.BACKFILL_PARTITION
    assert remediation.parameters["partition"] == "2026-09-20"


def test_unknown_schema_raises_rather_than_guessing(llm: HeuristicLLM) -> None:
    from pydantic import BaseModel

    from app.llm import LLMError

    class Unrelated(BaseModel):
        x: int = 0

    with pytest.raises(LLMError):
        llm.structured(messages_for(incident()), Unrelated)
