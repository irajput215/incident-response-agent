"""Prompt construction, one function per node.

Prompts live together because they are reviewed together — a change to what the
investigator is told changes what the root-cause stage can conclude, and reading
them side by side is the only way to keep them consistent.

Two conventions, both load-bearing:

**The incident is always embedded as a JSON object.** Prose summaries lose the
field names, and the field names are what the tools take as arguments. It also
means the deterministic analyst can parse the same prompts a model reads — the
offline path exercises the real prompt, not a parallel one.

**Evidence is rendered as numbered lines, including its structured detail.**
"Row count was zero" is a claim; the detail that produced it is the evidence. And
a conclusion is only as good as the fact that the agent can cite it.
"""
from __future__ import annotations

import json
from typing import Any

from app.llm.base import Message

TRIAGE_SYSTEM = """You are the triage stage of an automated incident-response agent for a \
data platform. A scheduled pipeline has failed and you must classify the failure \
quickly and cheaply, before any expensive investigation.

Classify the failure into exactly one category, assign a severity, and state a \
first hypothesis for the investigator to confirm or discard.

Available categories (use the exact string):
  DATA_SOURCE_FAILURE          an expected input file, key or partition is missing
  UPSTREAM_DEPENDENCY_FAILURE  a pipeline this one depends on failed or never delivered
  SCHEMA_CHANGE                an upstream schema changed and the job did not adapt
  DATA_QUALITY_FAILURE         a data-quality assertion rejected the output
  RESOURCE_EXHAUSTION          a memory, disk or executor limit was exceeded
  CONFIGURATION_ERROR          configuration, credentials or permissions are wrong
  TRANSIENT_INFRASTRUCTURE     a connection, timeout or transient infrastructure fault
  CODE_ERROR                   a defect in the pipeline code raised an exception
  UNKNOWN                      the text does not identify a cause

Severity is one of CRITICAL, HIGH, MEDIUM, LOW. Weigh business impact: a \
criticality-high pipeline that produced no data is HIGH or CRITICAL.

The confidence field is your own assessment, not a calculation. Be honest: if the \
error text is ambiguous, say so with a low number. A confidently wrong triage \
costs more than an uncertain one."""

INVESTIGATION_SYSTEM = """You are the investigation stage of an automated incident-response \
agent for a data platform. Triage has formed a first hypothesis. Your job is to \
gather evidence that confirms or refutes it.

You have tools. Use them — do not speculate about data you have not looked at. \
Reason from what the tools return, and prefer evidence that discriminates between \
competing explanations over evidence that merely restates the failure.

Work from the failure outward:
  1. Read what the failed run actually reported.
  2. Establish whether the expected input arrived, and what the data looks like now.
  3. Compare against a run that succeeded. What is different?
  4. Check the pipeline's dependencies. If something this pipeline depends on also \
failed, that is usually the cause, and rerunning this pipeline will not help.

Call tools until you can explain the failure, then stop and reply with a short \
summary of what you found. If the evidence is genuinely insufficient, say so \
plainly rather than guessing — an honest "I cannot tell yet" is a useful result."""

ROOT_CAUSE_SYSTEM = """You are the root-cause stage of an automated incident-response agent \
for a data platform. You are given the incident, the triage hypothesis, and all \
evidence the investigator collected.

Decide whether the evidence establishes a single root cause.

Set is_conclusive to true only when the evidence actually supports one explanation. \
If the evidence is thin, contradictory, or consistent with several causes, set \
is_conclusive to false — the workflow will gather more evidence, and a confident \
wrong answer is far more damaging than an explicit "not yet".

Confidence is your own honest assessment of how well the evidence supports the \
conclusion. Do not compute it from the number of signals; judge it. If you are \
conclusive, list the specific observations that support the conclusion in the \
evidence array, quoting the numbers you were given."""

REMEDIATION_SYSTEM = """You are the remediation stage of an automated incident-response agent \
for a data platform. A root cause has been established. Propose the action that \
actually fixes it.

Choose exactly one action:
  RERUN_PIPELINE        rerun the failed pipeline (the fault was transient or its input is now present)
  RERUN_UPSTREAM        rerun the upstream pipeline that failed, then this one
  BACKFILL_PARTITION    repopulate a specific missing partition
  ALERT_OWNER           notify the owning team; the fix needs a human or another team
  ESCALATE              hand to a human investigator with full context
  NO_ACTION             record the incident without further action

Reason about *where the fault actually is*. If an upstream pipeline failed, \
rerunning the downstream pipeline will simply fail again — the target must be the \
upstream pipeline. Set requires_approval to true for any action that changes data \
or triggers work; the system will not execute anything without a human for those.

Be explicit about the risk and about what you would do if the action does not work."""


def _json_block(payload: Any) -> str:
    # ensure_ascii=False so punctuation in a real error message (em dashes,
    # quotes) reaches the model — and the report — as written, not as \u2014.
    return json.dumps(payload, indent=2, default=str, ensure_ascii=False)


def _evidence_block(evidence: list[dict[str, Any]], limit: int = 40) -> str:
    """Evidence as numbered findings, each with its structured detail."""
    if not evidence:
        return "(no evidence collected yet)"
    lines: list[str] = []
    for index, item in enumerate(evidence[:limit], start=1):
        lines.append(
            f"{index}. [{item.get('source', '?')}:{item.get('tool', '?')}] {item.get('summary', '')}"
        )
        detail = item.get("detail")
        if detail:
            lines.append(f"   detail: {json.dumps(detail, default=str, ensure_ascii=False)}")
        if item.get("supports"):
            lines.append(f"   supports: {item['supports']}")
        if item.get("refutes"):
            lines.append(f"   refutes: {item['refutes']}")
    if len(evidence) > limit:
        lines.append(f"... and {len(evidence) - limit} more observation(s)")
    return "\n".join(lines)


def triage_messages(incident: dict[str, Any]) -> list[Message]:
    return [
        {"role": "system", "content": TRIAGE_SYSTEM},
        {
            "role": "user",
            "content": (
                "A pipeline run has failed. Triage it.\n\n"
                f"INCIDENT:\n{_json_block(incident)}"
            ),
        },
    ]


def investigation_messages(
    incident: dict[str, Any],
    triage: dict[str, Any] | None,
    evidence: list[dict[str, Any]],
    *,
    round_number: int = 1,
) -> list[Message]:
    hypothesis = (triage or {}).get("initial_hypothesis", "none recorded")
    category = (triage or {}).get("category", "UNKNOWN")
    return [
        {"role": "system", "content": INVESTIGATION_SYSTEM},
        {
            "role": "user",
            "content": (
                f"INCIDENT:\n{_json_block(incident)}\n\n"
                f"TRIAGE: category={category}, hypothesis={hypothesis!r}\n\n"
                f"EVIDENCE COLLECTED SO FAR (round {round_number}):\n"
                f"{_evidence_block(evidence)}\n\n"
                "Gather the evidence needed to confirm or refute the hypothesis. "
                "Call a tool, or state your findings if you have enough."
            ),
        },
    ]


def root_cause_messages(
    incident: dict[str, Any],
    triage: dict[str, Any] | None,
    evidence: list[dict[str, Any]],
) -> list[Message]:
    return [
        {"role": "system", "content": ROOT_CAUSE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"INCIDENT:\n{_json_block(incident)}\n\n"
                f"TRIAGE:\n{_json_block(triage or {})}\n\n"
                f"EVIDENCE:\n{_evidence_block(evidence)}\n\n"
                "Establish the root cause, or state that the evidence is insufficient."
            ),
        },
    ]


def remediation_messages(
    incident: dict[str, Any],
    root_cause: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> list[Message]:
    return [
        {"role": "system", "content": REMEDIATION_SYSTEM},
        {
            "role": "user",
            "content": (
                f"INCIDENT:\n{_json_block(incident)}\n\n"
                f"ROOT CAUSE:\n{_json_block(root_cause)}\n\n"
                f"EVIDENCE:\n{_evidence_block(evidence)}\n\n"
                "Propose the remediation."
            ),
        },
    ]


__all__ = [
    "investigation_messages",
    "remediation_messages",
    "root_cause_messages",
    "triage_messages",
]
