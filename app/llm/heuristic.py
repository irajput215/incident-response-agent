"""A deterministic, offline "analyst" that satisfies the :class:`LLMClient` protocol.

This is **not a mock**. A mock returns whatever a test tells it to, which makes
the test tautological. This is a *rule-based baseline*: a real keyword/regex
classifier that reads the same prompts a model would and produces the same
schema-shaped objects using explicit signals.

Two consequences, both deliberate:

1. **CI becomes meaningful.** The evaluation suite can score root-cause accuracy
   with no API key and no network, and the number it reports is a real number
   about a real (if simple) system.
2. **You get a baseline to beat.** "The LLM scored 0.86 on root-cause accuracy"
   means nothing on its own. "The heuristic baseline scores 0.62 and the model
   scores 0.86" is a finding — and where the two disagree is exactly where the
   model is earning its cost.

⚠️ Its limits are real and worth stating: it has no notion of meaning, so it
fails on any failure expressed in unfamiliar vocabulary, and its severity rule
is a lookup table rather than an impact assessment.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from app.config import STUB_MODEL
from app.llm.base import (
    LLMError,
    LLMResponse,
    Message,
    TModel,
    ToolCall,
    ToolSpec,
    estimate_tokens,
)
from app.schemas import (
    EvidenceSource,
    FailureCategory,
    Remediation,
    RemediationAction,
    RiskLevel,
    RootCause,
    Severity,
    Triage,
)

# Ordered: the first category whose signal appears wins. Order encodes priority —
# a log that says both "file not found" and "upstream task failed" is an upstream
# problem, and putting UPSTREAM_DEPENDENCY_FAILURE late would misclassify it.
_SIGNALS: tuple[tuple[FailureCategory, tuple[str, ...]], ...] = (
    (
        FailureCategory.SCHEMA_CHANGE,
        (
            "analysisexception",
            "cannot resolve",
            "could not resolve",
            "column not found",
            "no such column",
            "missing column",
            "unexpected column",
            "schema mismatch",
            "schema drift",
            "cannot cast",
            "type mismatch",
            "failed to parse column",
            "unknown field",
        ),
    ),
    (
        FailureCategory.RESOURCE_EXHAUSTION,
        (
            "outofmemory",
            "out of memory",
            "oom",
            "memory limit",
            "executor lost",
            "container killed",
            "killed by yarn",
            "no space left",
            "disk full",
            "exceeds available memory",
            "too many open files",
        ),
    ),
    (
        FailureCategory.CONFIGURATION_ERROR,
        (
            "configurationexception",
            "invalid configuration",
            "missing required config",
            "environment variable",
            "accessdenied",
            "access denied",
            "permission denied",
            "invalid credentials",
            "unknown option",
            "not authorized",
        ),
    ),
    (
        FailureCategory.UPSTREAM_DEPENDENCY_FAILURE,
        (
            "upstream task failed",
            "upstream failed",
            "upstream pipeline",
            "upstream dependency",
            "upstream run failed",
            "parent task failed",
            "previous task failed",
            "trigger rule",
            "dag run failed",
            "dependency not met",
        ),
    ),
    (
        FailureCategory.DATA_SOURCE_FAILURE,
        (
            "file not found",
            "filenotfound",
            "no such file",
            "nosuchkey",
            "path does not exist",
            "partition not found",
            "missing partition",
            "no files to process",
            "empty directory",
            "does not exist",
            "key does not exist",
            "source unavailable",
        ),
    ),
    (
        FailureCategory.TRANSIENT_INFRASTRUCTURE,
        (
            "connection reset",
            "connection refused",
            "could not connect",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "service unavailable",
            "too many connections",
            "deadlock detected",
            "503",
            "502",
        ),
    ),
    (
        FailureCategory.DATA_QUALITY_FAILURE,
        (
            "null rate",
            "null_rate",
            "duplicate key",
            "duplicate rows",
            "referential integrity",
            "freshness",
            "row count mismatch",
            "unexpected row count",
            "assertion failed",
            "quality check failed",
            "not-null",
        ),
    ),
    (
        FailureCategory.CODE_ERROR,
        (
            "typeerror",
            "valueerror",
            "keyerror",
            "attributeerror",
            "importerror",
            "syntaxerror",
            "nameerror",
            "indexerror",
        ),
    ),
)

# Baseline severity policy: a lookup on category, with two error-text overrides.
# ⚠️ This is a policy table, not an impact analysis — it cannot know that one
# pipeline is revenue-critical and another is a nightly report.
_SEVERITY_BY_CATEGORY: dict[FailureCategory, Severity] = {
    FailureCategory.DATA_SOURCE_FAILURE: Severity.HIGH,
    FailureCategory.UPSTREAM_DEPENDENCY_FAILURE: Severity.HIGH,
    FailureCategory.SCHEMA_CHANGE: Severity.HIGH,
    FailureCategory.DATA_QUALITY_FAILURE: Severity.HIGH,
    FailureCategory.RESOURCE_EXHAUSTION: Severity.HIGH,
    FailureCategory.CONFIGURATION_ERROR: Severity.HIGH,
    FailureCategory.TRANSIENT_INFRASTRUCTURE: Severity.MEDIUM,
    FailureCategory.CODE_ERROR: Severity.HIGH,
    FailureCategory.UNKNOWN: Severity.MEDIUM,
}

_REMEDIATION_BY_CATEGORY: dict[FailureCategory, tuple[RemediationAction, RiskLevel, bool]] = {
    FailureCategory.UPSTREAM_DEPENDENCY_FAILURE: (
        RemediationAction.RERUN_UPSTREAM,
        RiskLevel.MEDIUM,
        True,
    ),
    FailureCategory.DATA_SOURCE_FAILURE: (
        RemediationAction.BACKFILL_PARTITION,
        RiskLevel.MEDIUM,
        True,
    ),
    FailureCategory.TRANSIENT_INFRASTRUCTURE: (
        RemediationAction.RERUN_PIPELINE,
        RiskLevel.LOW,
        True,
    ),
    FailureCategory.RESOURCE_EXHAUSTION: (
        RemediationAction.RERUN_PIPELINE,
        RiskLevel.MEDIUM,
        True,
    ),
    FailureCategory.SCHEMA_CHANGE: (RemediationAction.ALERT_OWNER, RiskLevel.LOW, False),
    FailureCategory.CONFIGURATION_ERROR: (RemediationAction.ALERT_OWNER, RiskLevel.LOW, False),
    FailureCategory.DATA_QUALITY_FAILURE: (RemediationAction.ESCALATE, RiskLevel.LOW, False),
    FailureCategory.CODE_ERROR: (RemediationAction.ESCALATE, RiskLevel.LOW, False),
    FailureCategory.UNKNOWN: (RemediationAction.ESCALATE, RiskLevel.LOW, False),
}

# The investigation plan this baseline replays, in order. Filtered against the
# tools actually offered, so adding or removing a real tool never leaves this
# list pointing at something that does not exist.
#
# `get_pipeline_metadata` sits in the middle on purpose: its result is what makes
# the dependency follow-up below possible, and without it the baseline can never
# discover that an *upstream* pipeline is the actual cause.
_INVESTIGATION_PLAN: tuple[str, ...] = (
    "get_pipeline_logs",
    "get_previous_runs",
    "get_pipeline_metadata",
    "check_latest_partition",
    "check_row_count",
    "check_table_schema",
    "search_logs",
)

# Two dynamic follow-ups run after the static plan, each driven by evidence the
# plan just produced. Without them the baseline collects facts but never joins
# them, which is the difference between a log scraper and an investigator.
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# The evidence block renders one line per observation as `[source:tool] summary`.
# Matching those markers is how the analyst knows what it has already asked,
# which matters because a *later investigation round* is handed a freshly built
# message list containing the evidence as text — not the original tool messages.
_EVIDENCE_MARKER_RE = re.compile(r"\[[a-z_]+:([a-z_]+)\]")


def _tools_with_evidence(text: str) -> set[str]:
    """Tool names that have already contributed an observation."""
    return set(_EVIDENCE_MARKER_RE.findall(text))


def _partition_from(incident: dict[str, Any]) -> str:
    """The partition the failure is about, read out of the alert's own text."""
    for key in ("partition", "date", "error", "logs", "message"):
        match = _DATE_RE.search(str(incident.get(key) or ""))
        if match:
            return match.group(1)
    return ""

# --- upstream-dependency signals -------------------------------------------
# The gap between "upstream" and the failure verb must tolerate JSON
# punctuation, because a tool result looks like
#   "upstream_pipeline": "claims_ingestion", "status": "FAILED"
# and a gap class of only \w\s_- stops at the first quote. It must NOT tolerate
# "." or newlines, or the pattern would happily span two unrelated sentences.
# `[ \t]` rather than `\s` is what keeps the match inside one line.
_GAP = r'[\w \t_\-":,]{0,60}?'
_VERB = r"(fail|error|missing|not\s+found|absent|zero|no\s+data|unavailable|did\s+not\s+complete)"

_UPSTREAM_FAILURE_RE = re.compile(rf"upstream{_GAP}{_VERB}", re.IGNORECASE)
_PREDECESSOR_FAILURE_RE = re.compile(
    rf"(parent|previous|predecessor){_GAP}task{_GAP}fail", re.IGNORECASE
)

# Regex signals, checked before the literal ones. "upstream ingestion failed" is
# the same finding as "upstream failed", and a literal substring list can never
# enumerate every word a human might put in between. Matching the *shape*
# ("upstream <anything> failed") is what makes this robust — a literal-only
# version misclassified this exact case as a data-source failure.
_CATEGORY_REGEXES: tuple[tuple[FailureCategory, tuple[re.Pattern[str], ...]], ...] = (
    (
        FailureCategory.UPSTREAM_DEPENDENCY_FAILURE,
        (_UPSTREAM_FAILURE_RE, _PREDECESSOR_FAILURE_RE),
    ),
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def detect_category(text: str) -> FailureCategory:
    """First matching signal wins. Returns ``UNKNOWN`` when nothing matches."""
    for regex_category, regexes in _CATEGORY_REGEXES:
        for regex in regexes:
            if regex.search(text):
                return regex_category
    normalized = _normalize(text)
    for literal_category, literals in _SIGNALS:
        for literal in literals:
            if literal in normalized:
                return literal_category
    return FailureCategory.UNKNOWN


def detect_severity(text: str, category: FailureCategory) -> Severity:
    normalized = _normalize(text)
    if any(w in normalized for w in ("data loss", "corruption", "corrupted", "pii leak")):
        return Severity.CRITICAL
    if "degraded" in normalized and category is FailureCategory.TRANSIENT_INFRASTRUCTURE:
        return Severity.LOW
    return _SEVERITY_BY_CATEGORY.get(category, Severity.MEDIUM)


def _message_text(messages: list[Message]) -> str:
    """Flatten every message into one searchable string.

    Tool results are included: the root-cause prompt carries the collected
    evidence, and the baseline has to read it to reason about it.
    """
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
        calls = message.get("tool_calls")
        if calls:
            parts.append(json.dumps(calls, default=str))
    return "\n".join(parts)


def _first_incident_json(text: str) -> dict[str, Any]:
    """The embedded incident payload, or ``{}`` when there is not one."""
    return _first_json_with_key(text, "pipeline") or {}


def _first_json_with_key(text: str, key: str) -> dict[str, Any] | None:
    """First balanced ``{...}`` object in ``text`` that contains ``key``.

    Brace-counting rather than a regex, and string-aware so a brace inside a
    string literal does not unbalance the scan.
    """
    for match in re.finditer(r"\{", text):
        depth = 0
        in_string = False
        escaped = False
        for index in range(match.start(), len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[match.start() : index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict) and key in parsed:
                        return parsed
                    break
    return None


def _tool_call_history(messages: list[Message]) -> list[tuple[str, dict[str, Any]]]:
    """Every tool call the assistant has already requested, in order.

    Needed to know what has been asked *about*, which is different from knowing
    how many results came back — a failed call still means "already tried".
    """
    history: list[tuple[str, dict[str, Any]]] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = (call or {}).get("function") or {}
            name = function.get("name")
            raw = function.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                args = {}
            if name:
                history.append((str(name), args if isinstance(args, dict) else {}))
    return history


def _tool_result_data(message: Message) -> dict[str, Any]:
    """The ``data`` payload of a tool message, or ``{}``."""
    content = message.get("content")
    if not isinstance(content, str):
        return {}
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def _json_objects_in(text: str) -> list[dict[str, Any]]:
    """Every top-level balanced ``{...}`` object in ``text``.

    Nested objects are consumed by their parent's balanced scan, so this does
    not return duplicates.
    """
    objects: list[dict[str, Any]] = []
    index = 0
    while True:
        start = text.find("{", index)
        if start == -1:
            return objects
        depth = 0
        in_string = False
        escaped = False
        end: int | None = None
        for position in range(start, len(text)):
            char = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = position
                    break
        if end is None:
            index = start + 1
            continue
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                objects.append(parsed)
        except json.JSONDecodeError:
            pass
        index = end + 1


def _evidence_payloads(messages: list[Message]) -> list[dict[str, Any]]:
    """Structured tool-result data the analyst can see, from either representation.

    During the investigation loop the results arrive as real ``role: "tool"``
    messages. In the **root-cause and remediation prompts the same evidence has
    already been rendered into the user message as text** — so an analyst that
    only looks for tool messages finds nothing and silently falls back to
    keyword matching, which is exactly the bug this function fixes. Both paths
    matter: the first is what the agent sees while gathering, the second is what
    it sees while concluding.
    """
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "tool":
            data = _tool_result_data(message)
            if data:
                payloads.append(data)
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for obj in _json_objects_in(content):
            nested = obj.get("data")
            payloads.append(nested if isinstance(nested, dict) else obj)
    return payloads


def _upstream_pipelines(payloads: list[dict[str, Any]]) -> list[str]:
    """Upstream pipeline names discovered in any evidence payload."""
    found: list[str] = []
    for data in payloads:
        upstream = data.get("upstream")
        if isinstance(upstream, list):
            for name in upstream:
                if isinstance(name, str) and name and name not in found:
                    found.append(name)
        single = data.get("upstream_pipeline")
        if isinstance(single, str) and single and single not in found:
            found.append(single)
    return found


def _upstream_failure_in_evidence(payloads: list[dict[str, Any]]) -> str | None:
    """The upstream pipeline that *failed*, if the evidence shows one.

    This is the two-step inference the scenario turns on, and it cannot be done
    with a regex over the prompt: "this job depends on X" arrives in one tool
    result and "X failed" in a different one. Joining them requires reading both
    structurally.
    """
    upstream = _upstream_pipelines(payloads)
    if not upstream:
        return None
    for data in payloads:
        pipeline = data.get("pipeline")
        if pipeline not in upstream:
            continue
        for run in data.get("runs") or []:
            if isinstance(run, dict) and str(run.get("status", "")).upper() == "FAILED":
                return str(pipeline)
    return None


def _schema_column_matching_error(
    payloads: list[dict[str, Any]], incident: dict[str, Any]
) -> tuple[str, str] | None:
    """A column that appears both in the table's schema and in the alert's error.

    This is how a null-rate assertion gets corroborated with a measured number
    instead of being taken on trust: the error names the column, the schema tool
    confirms it exists, and the null-rate tool measures it.
    """
    error = str(incident.get("error") or "").lower()
    if not error:
        return None
    for data in payloads:
        columns = data.get("columns")
        if not isinstance(columns, list):
            continue
        table = str(data.get("table") or "").split(".")[-1]
        for column in columns:
            name = column.get("column_name") if isinstance(column, dict) else None
            if isinstance(name, str) and name and name.lower() in error:
                return table, name
    return None


class HeuristicLLM:
    """Offline, deterministic implementation of :class:`LLMClient`."""

    def __init__(self, model: str = STUB_MODEL) -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    # --- text / tool-calling turn -----------------------------------------
    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        text = _message_text(messages)
        prompt_tokens = estimate_tokens(text)

        if tools:
            call = self._next_tool_call(messages, tools)
            if call is not None:
                return LLMResponse(
                    text="",
                    tool_calls=[call],
                    model=self._model,
                    tokens_in=prompt_tokens,
                    tokens_out=estimate_tokens(json.dumps(call.arguments)),
                    latency_ms=0.0,
                )
            summary = "Evidence collection complete for the available tool set."
            return LLMResponse(
                text=summary,
                model=self._model,
                tokens_in=prompt_tokens,
                tokens_out=estimate_tokens(summary),
                latency_ms=0.0,
            )

        reply = "Offline heuristic analyst: no tools offered, no text task recognised."
        return LLMResponse(
            text=reply,
            model=self._model,
            tokens_in=prompt_tokens,
            tokens_out=estimate_tokens(reply),
            latency_ms=0.0,
        )

    def _next_tool_call(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> ToolCall | None:
        """Replay the next step of the investigation plan.

        Progress is derived from the conversation itself — which observations are
        already present and which calls have been made — rather than from shared
        mutable state, so concurrent investigations cannot interfere.

        Measured against the *evidence*, not against how many tool messages are
        present. A later round is handed a rebuilt message list in which the
        evidence appears as text, so counting tool messages would reset the plan
        and re-ask every question — burning the call budget on a round that
        already has the answers.
        """
        available = {tool.name for tool in tools}
        text = _message_text(messages)
        payloads = _evidence_payloads(messages)
        incident = _first_incident_json(text)

        done = _tools_with_evidence(text) | {
            name for name, _ in _tool_call_history(messages)
        }
        step = len(done)

        for name in _INVESTIGATION_PLAN:
            if name in available and name not in done:
                return ToolCall(
                    id=f"heuristic-{step}-{name}",
                    name=name,
                    arguments=self._arguments_for(name, incident),
                )

        # Follow-up 1: corroborate a data-quality claim with a measured number,
        # when the alert's own text names a column the schema also has.
        if "check_null_rate" in available:
            already_measured = {
                (str(args.get("table")), str(args.get("column")))
                for name, args in _tool_call_history(messages)
                if name == "check_null_rate"
            }
            candidate = _schema_column_matching_error(payloads, incident)
            if candidate and candidate not in already_measured:
                table, column = candidate
                arguments: dict[str, Any] = {"table": table, "column": column}
                part = _partition_from(incident)
                if part:
                    arguments["partition"] = part
                return ToolCall(
                    id=f"heuristic-{step}-null-rate-{column}",
                    name="check_null_rate",
                    arguments=arguments,
                )

        # Follow-up 2: follow the dependency chain. An upstream pipeline that
        # also failed is the single most valuable fact available, and it never
        # appears in this pipeline's own logs. A real model reaches this by
        # reading the metadata result; the baseline reads the same result.
        if "get_previous_runs" in available:
            already_asked = {
                str(args.get("pipeline"))
                for name, args in _tool_call_history(messages)
                if name == "get_previous_runs"
            }
            for upstream in _upstream_pipelines(payloads):
                if upstream not in already_asked:
                    return ToolCall(
                        id=f"heuristic-{step}-upstream-{upstream}",
                        name="get_previous_runs",
                        arguments={"pipeline": upstream, "limit": 5},
                    )

        return None

    def _arguments_for(self, name: str, incident: dict[str, Any]) -> dict[str, Any]:
        pipeline = str(incident.get("pipeline") or "")
        run_id = str(incident.get("run_id") or "")
        table = str(incident.get("table") or pipeline.removesuffix("_daily"))
        error = str(incident.get("error") or "")
        partition = _partition_from(incident)

        if name == "get_pipeline_logs":
            return {"pipeline": pipeline, "run_id": run_id}
        if name == "get_previous_runs":
            return {"pipeline": pipeline, "limit": 5}
        if name == "get_pipeline_metadata":
            return {"pipeline": pipeline}
        if name == "search_logs":
            # Search for the most distinctive word in the error, so the baseline
            # demonstrates the search tool rather than re-fetching everything.
            pattern = next(
                (w for w in re.findall(r"[A-Za-z]{5,}", error) if w.lower() not in _STOPWORDS),
                "ERROR",
            )
            return {"pipeline": pipeline, "pattern": pattern}
        if name == "check_latest_partition":
            return {"table": table}
        if name == "check_table_schema":
            return {"table": table}
        if name == "check_row_count":
            # Passing the partition is what turns this from "here are the
            # partitions" into "the partition you care about is EMPTY, and the
            # one before it had 200 rows" — the observation the whole
            # missing-input diagnosis rests on.
            return {"table": table, "partition": partition} if partition else {"table": table}
        if name == "check_null_rate":
            column = str(incident.get("column") or "")
            args: dict[str, Any] = {"table": table, "column": column}
            if partition:
                args["partition"] = partition
            return args
        return {}

    # --- structured turn ---------------------------------------------------
    def structured(
        self,
        messages: list[Message],
        schema: type[TModel],
        *,
        temperature: float | None = None,
    ) -> TModel:
        text = _message_text(messages)
        tokens_in = estimate_tokens(text)

        if schema is Triage:
            result: BaseModel = self._triage(messages)
        elif schema is RootCause:
            result = self._root_cause(messages)
        elif schema is Remediation:
            result = self._remediation(messages)
        else:
            raise LLMError(f"heuristic analyst has no rule for schema {schema.__name__}")

        # Re-validated through the requested schema so the return type is the
        # caller's type, not the loose BaseModel the rules build internally.
        _ = tokens_in
        return schema.model_validate(result.model_dump())

    # --- the rules ---------------------------------------------------------
    def _triage(self, messages: list[Message]) -> Triage:
        text = _message_text(messages)
        incident = _first_incident_json(text)
        evidence_text = " ".join(
            str(incident.get(k) or "") for k in ("error", "logs", "message")
        ) or text

        category = detect_category(evidence_text)
        severity = detect_severity(evidence_text, category)

        if category is FailureCategory.UNKNOWN:
            hypothesis = "Insufficient signal in the failure text to form a hypothesis"
            rationale = "No recognised failure signature matched; needs investigation."
            confidence = 0.2
        else:
            hypothesis = _HYPOTHESES[category]
            rationale = f"Matched a {category.value} signature in the failure text."
            confidence = 0.65

        return Triage(
            category=category,
            severity=severity,
            initial_hypothesis=hypothesis,
            rationale=rationale,
            confidence=confidence,
        )

    def _root_cause(self, messages: list[Message]) -> RootCause:
        """Conclude only when the evidence supports it.

        The reasoning order matters and mirrors how an engineer actually works:

        1. **What does this job's own error say?** If it names a specific
           mechanism — an OOM kill, a missing column, a refused connection — that
           is direct evidence and it wins. Circumstantial evidence should not
           override a stack trace.
        2. **If the error only says "my input was not there", ask who was
           supposed to provide it.** A failed upstream pipeline is the answer,
           and it is invisible in this job's own logs — which is exactly why
           step 2 needs the metadata and run-history tools.
        3. Otherwise fall back to the whole evidence stream, and if even that
           yields nothing, say so with ``is_conclusive=False`` rather than
           guessing. That refusal is what drives the graph's next round.
        """
        text = _message_text(messages)
        payloads = _evidence_payloads(messages)
        incident = _first_incident_json(text)

        direct_text = " ".join(
            str(incident.get(key) or "") for key in ("error", "logs", "message")
        )
        direct = detect_category(direct_text) if direct_text.strip() else detect_category(text)

        # Only chase the dependency chain when the direct evidence does not
        # already name a mechanism. `detect_category` returns "input missing"
        # for a missing file, which is precisely the case an upstream failure
        # explains — and precisely the case where rerunning this job is useless.
        upstream_failure = (
            _upstream_failure_in_evidence(payloads)
            if direct in _DEPENDENCY_INDICATING_CATEGORIES
            else None
        )

        if upstream_failure:
            category = FailureCategory.UPSTREAM_DEPENDENCY_FAILURE
            confidence = 0.9
        elif direct is not FailureCategory.UNKNOWN:
            category = direct
            confidence = 0.7
        else:
            category = detect_category(text)
            if category is FailureCategory.UNKNOWN:
                return RootCause(
                    root_cause="No single root cause could be established from the evidence",
                    category=FailureCategory.UNKNOWN,
                    confidence=0.2,
                    is_conclusive=False,
                    evidence=[],
                    recommended_action="Gather more evidence or escalate to a human",
                    reasoning="No recognised failure signature appeared in the collected evidence.",
                )
            confidence = 0.65

        evidence = _extract_evidence_lines(text)
        return RootCause(
            root_cause=_ROOT_CAUSES[category],
            category=category,
            confidence=confidence,
            is_conclusive=True,
            evidence=evidence,
            recommended_action=_RECOMMENDATIONS[category],
            reasoning=(
                f"Classified as {category.value} from the failed run's own error text"
                if category is direct
                else "Upstream dependency "
                f"{upstream_failure!r} failed for the same partition, which explains "
                "the missing input; the failed job's own logs cannot show this."
            ),
        )

    def _remediation(self, messages: list[Message]) -> Remediation:
        """Choose the fix from the *established root cause*, not the whole prompt.

        Re-classifying the entire prompt was a real bug: the evidence stream
        contains every pipeline's metadata, including its ``upstream`` list, so a
        whole-text match reported "upstream dependency" for every incident and
        proposed rerunning an upstream pipeline to fix an out-of-memory error.
        The root cause is already stated in the prompt; read that.
        """
        text = _message_text(messages)
        payloads = _evidence_payloads(messages)

        root = _first_json_with_key(text, "root_cause") or {}
        raw_category = root.get("category")
        known = {c.value for c in FailureCategory}
        if isinstance(raw_category, str) and raw_category in known:
            category = FailureCategory(raw_category)
        else:
            category = detect_category(text)

        action, risk, requires_approval = _REMEDIATION_BY_CATEGORY.get(
            category, (RemediationAction.ESCALATE, RiskLevel.LOW, False)
        )
        incident = _first_incident_json(text)
        pipeline = str(incident.get("pipeline") or "unknown_pipeline")

        target = pipeline
        parameters: dict[str, Any] = {}
        if action is RemediationAction.RERUN_UPSTREAM:
            # The fix belongs to the upstream pipeline that failed. Rerunning the
            # failed job would just fail again, because its input is still
            # missing — so recovering the right name is load-bearing.
            target = (
                _upstream_failure_in_evidence(payloads)
                or _first_string_field(text, "upstream_pipeline", default=pipeline)
            )
        elif action is RemediationAction.BACKFILL_PARTITION:
            parameters = {
                "partition": _first_string_field(
                    text,
                    "expected_partition",
                    default=_first_string_field(text, "partition", default=""),
                )
            }

        return Remediation(
            action=action,
            target=target,
            parameters=parameters,
            risk=risk,
            requires_approval=requires_approval,
            rationale=f"Standard remediation for a {category.value} finding.",
            expected_effect=_EXPECTED_EFFECTS[action],
            rollback="Re-run the original pipeline definition; no destructive step is taken.",
        )


_STOPWORDS = frozenset({"error", "failed", "failure", "exception", "cannot", "could", "traceback"})

# Categories that mean "my input was not there". These are the only ones worth
# chasing up the dependency chain: if the error already names a mechanism (an OOM
# kill, a missing column), a failed upstream pipeline is a coincidence, not the
# cause — and blaming it would send the fix to the wrong team.
_DEPENDENCY_INDICATING_CATEGORIES = frozenset(
    {
        FailureCategory.DATA_SOURCE_FAILURE,
        FailureCategory.UPSTREAM_DEPENDENCY_FAILURE,
        FailureCategory.UNKNOWN,
    }
)

_HYPOTHESES: dict[FailureCategory, str] = {
    FailureCategory.DATA_SOURCE_FAILURE: "An expected input file or partition is missing",
    FailureCategory.UPSTREAM_DEPENDENCY_FAILURE: "An upstream pipeline did not deliver its output",
    FailureCategory.SCHEMA_CHANGE: "An upstream schema changed and the job did not adapt",
    FailureCategory.DATA_QUALITY_FAILURE: "A data-quality assertion rejected the output",
    FailureCategory.RESOURCE_EXHAUSTION: "The job exceeded a memory, disk or executor limit",
    FailureCategory.CONFIGURATION_ERROR: "Configuration, credentials or permissions are wrong",
    FailureCategory.TRANSIENT_INFRASTRUCTURE: "A transient infrastructure fault interrupted the run",
    FailureCategory.CODE_ERROR: "A defect in the pipeline code raised an exception",
    FailureCategory.UNKNOWN: "Unknown failure mode",
}

_ROOT_CAUSES: dict[FailureCategory, str] = {
    FailureCategory.DATA_SOURCE_FAILURE: (
        "The expected input partition was never delivered to the landing zone"
    ),
    FailureCategory.UPSTREAM_DEPENDENCY_FAILURE: (
        "The upstream ingestion pipeline failed, so the expected partition was never written"
    ),
    FailureCategory.SCHEMA_CHANGE: (
        "An upstream schema change removed or renamed a column this job depends on"
    ),
    FailureCategory.DATA_QUALITY_FAILURE: (
        "A data-quality check rejected the output, most likely because of unexpected nulls or duplicates"
    ),
    FailureCategory.RESOURCE_EXHAUSTION: (
        "The job exceeded its allocated memory and was killed by the executor"
    ),
    FailureCategory.CONFIGURATION_ERROR: (
        "The job ran with invalid configuration, credentials or permissions"
    ),
    FailureCategory.TRANSIENT_INFRASTRUCTURE: (
        "A transient infrastructure fault (connection or dependency timeout) interrupted the run"
    ),
    FailureCategory.CODE_ERROR: "A defect in the pipeline code raised an unhandled exception",
    FailureCategory.UNKNOWN: "Root cause not established",
}

_RECOMMENDATIONS: dict[FailureCategory, str] = {
    FailureCategory.DATA_SOURCE_FAILURE: "Backfill the missing partition once the source is restored",
    FailureCategory.UPSTREAM_DEPENDENCY_FAILURE: "Rerun the upstream ingestion pipeline, then this job",
    FailureCategory.SCHEMA_CHANGE: "Alert the owning team and pin the upstream schema contract",
    FailureCategory.DATA_QUALITY_FAILURE: "Escalate to the data owner to review the quality assertion",
    FailureCategory.RESOURCE_EXHAUSTION: "Rerun with a larger memory allocation",
    FailureCategory.CONFIGURATION_ERROR: "Alert the owning team to correct the configuration",
    FailureCategory.TRANSIENT_INFRASTRUCTURE: "Rerun the pipeline; the fault was transient",
    FailureCategory.CODE_ERROR: "Escalate to the pipeline owner with the stack trace",
    FailureCategory.UNKNOWN: "Escalate to a human investigator",
}

_EXPECTED_EFFECTS: dict[RemediationAction, str] = {
    RemediationAction.RERUN_PIPELINE: "The pipeline reprocesses its inputs and completes",
    RemediationAction.RERUN_UPSTREAM: "The missing upstream output is produced, unblocking this job",
    RemediationAction.BACKFILL_PARTITION: "The missing partition is populated from the corrected source",
    RemediationAction.ALERT_OWNER: "The owning team is notified and takes over",
    RemediationAction.ESCALATE: "A human investigator reviews the incident with full context",
    RemediationAction.NO_ACTION: "The incident is recorded without further action",
}

_EVIDENCE_HINTS = (
    "fail",
    "error",
    "missing",
    "zero",
    "no data",
    "not found",
    "null",
    "absent",
    "upstream",
)

# Keys whose *presence* makes a field evidence, even when the value looks
# innocuous. `row_count = 0` matters and so does `expected_partition`; without
# this, the two signals that actually crack the scenario get filtered out for
# not containing a failure word.
_NOTABLE_KEYS = (
    "partition",
    "count",
    "rows",
    "upstream",
    "null",
    "schema",
    "fresh",
    "fail",
    "error",
    "missing",
    "expected",
)

# Matches both `"key": "value"` and `"key": 123` — tool results are JSON, so
# numeric findings (row_count: 0) arrive unquoted and a string-only pattern
# silently loses them.
_KV_RE = re.compile(r'"([A-Za-z_]{3,30})"\s*:\s*(?:"([^"]{1,80})"|(-?[\d.]+))')

# A JSON field on its own indented line, as produced by `json.dumps(indent=2)`.
_JSON_FIELD_RE = re.compile(r'^"([A-Za-z_]{3,30})"\s*:\s*(.+?),?$')

# A rendered evidence-block line: `3. [pipeline_history:get_previous_runs] …`.
# The summary inside it is already a complete statement, so the numbering and
# the tool marker are stripped rather than quoted into the report.
_RENDERED_EVIDENCE_RE = re.compile(r"^\d+\.\s*\[[a-z_]+:[a-z_]+\]\s*(.+)$")

# Keys that are the agent's *own* reasoning rather than observations about the
# world. Quoting the triage hypothesis back as "evidence" makes the report look
# like it is citing something when it is only restating itself.
_NOT_EVIDENCE_KEYS = frozenset(
    {"rationale", "initial_hypothesis", "category", "confidence", "severity", "logs", "summary"}
)


def _is_noteworthy(fragment: str) -> bool:
    low = fragment.lower()
    return detect_category(fragment) is not FailureCategory.UNKNOWN or any(
        hint in low for hint in _EVIDENCE_HINTS
    )


def _is_notable_key(key: str) -> bool:
    low = key.lower()
    return any(token in low for token in _NOTABLE_KEYS)


def _first_string_field(text: str, field: str, default: str = "") -> str:
    """Read one scalar out of free text or embedded JSON.

    The heuristic has no structured view of prior tool results — it only sees
    the prompt — so it has to mine the pipeline name back out of the evidence.
    """
    quoted = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if quoted:
        return quoted.group(1)
    bare = re.search(rf"\b{re.escape(field)}\b\s*[=:]\s*([A-Za-z0-9_.\-/]+)", text, re.IGNORECASE)
    return bare.group(1) if bare else default


def _extract_evidence_lines(text: str, limit: int = 6) -> list[str]:
    """Pull short, quotable failure statements out of the collected evidence.

    The raw evidence stream mixes prose log lines, pretty-printed JSON (the
    incident and root-cause blocks) and compact JSON (tool results). Dumping any
    of those verbatim into a report's evidence list produces something
    unreadable, so all three are normalised to ``key = value`` statements —
    ``row_count = 0`` is evidence; ``{"row_count": 0, ...}`` is a data dump.
    """
    lines: list[str] = []

    def add(fragment: str) -> None:
        fragment = fragment.strip().strip(",").strip()
        if fragment.lower().endswith("= null"):
            return  # a null field is an absence, not an observation
        if 8 <= len(fragment) <= 240 and fragment not in lines:
            lines.append(fragment)

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Compact JSON on one line: mine it for signal-bearing key/value pairs.
        if line.startswith(("{", "[")):
            for match in _KV_RE.finditer(line):
                key = match.group(1)
                value = match.group(2) if match.group(2) is not None else match.group(3)
                if key.lower() in _NOT_EVIDENCE_KEYS:
                    continue
                if _is_noteworthy(str(value)) or _is_notable_key(key):
                    add(f"{key} = {value}")
            continue

        # Pretty-printed JSON: one field per line.
        field = _JSON_FIELD_RE.match(line)
        if field:
            key = field.group(1)
            value = field.group(2).strip().rstrip(",").strip('"')
            if key.lower() in _NOT_EVIDENCE_KEYS:
                continue
            if _is_notable_key(key) or _is_noteworthy(value):
                add(f"{key} = {value}")
            continue

        # A rendered observation from the evidence block: keep its summary.
        rendered = _RENDERED_EVIDENCE_RE.match(line)
        if rendered:
            add(rendered.group(1))
            if len(lines) >= limit:
                break
            continue

        if _is_noteworthy(line):
            add(line)

        if len(lines) >= limit:
            break

    return lines[:limit]


__all__ = [
    "EvidenceSource",
    "HeuristicLLM",
    "detect_category",
    "detect_severity",
]
