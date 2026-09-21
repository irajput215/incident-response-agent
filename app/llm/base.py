"""The LLM interface every caller depends on.

Deliberately narrow: two methods, plain dicts for messages, and a dataclass
response. Nothing in this module imports a provider SDK, so the agent code
cannot accidentally couple itself to one — swappability is enforced by the
import graph, not by good intentions.

The second method, :meth:`LLMClient.structured`, is the important one. Asking a
model for JSON and hoping is how LLM systems fail in production; the interface
makes "return a validated ``Triage``" a *contract* that every implementation
must satisfy, so each of them has to deal with the malformed-output case
explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

Message = dict[str, Any]
TModel = TypeVar("TModel", bound=BaseModel)


class LLMError(RuntimeError):
    """Any failure to obtain a usable response from a model.

    Callers may catch this to degrade (fall back to the heuristic analyst, mark
    the incident unresolved); they should never let it escape a graph node
    uncaught, because a broken model must not take down the workflow.
    """


@dataclass(slots=True)
class ToolSpec:
    """A tool as the model sees it: name, prose, and a JSON Schema.

    The schema *is* the model's API. A tool without one is a stringly-typed
    function call waiting to be given the wrong argument.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        """OpenAI/LiteLLM function-calling wire format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class ToolCall:
    """A model's request to run a tool.

    ``arguments`` is parsed JSON. If the model emitted unparseable arguments,
    the client stores ``{}`` and records the problem in ``parse_error`` so the
    tool layer can reject it as a validation failure rather than crashing the
    loop on a ``json.JSONDecodeError``.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None


@dataclass(slots=True)
class LLMResponse:
    """A model response plus the telemetry the evaluation cares about."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class LLMClient(Protocol):
    """What the agent needs from a model. Nothing more."""

    @property
    def model(self) -> str:
        """Identifier recorded on every run, so a report says what produced it."""
        ...

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """One turn. May return text, tool calls, or both."""
        ...

    def structured(
        self,
        messages: list[Message],
        schema: type[TModel],
        *,
        temperature: float | None = None,
    ) -> TModel:
        """One turn, constrained to ``schema`` and validated before returning.

        Implementations must raise :class:`LLMError` rather than returning an
        object that does not validate — the caller is entitled to assume a
        ``structured`` call either gives it a valid model or raises.
        """
        ...


# --------------------------------------------------------------------------- #
# Shared helpers for implementations
# --------------------------------------------------------------------------- #
def render_schema_instruction(schema: type[BaseModel]) -> str:
    """The prompt fragment that turns free text into a schema-shaped object.

    Sending the schema in the prompt (rather than relying on a provider-specific
    JSON mode) is what makes this work identically across Anthropic, OpenAI,
    DeepSeek and a local Ollama model. Provider JSON modes are used as an
    *additional* hint where available, never as the only mechanism.
    """
    import json

    return (
        "Respond with a SINGLE JSON object and nothing else — no prose, no "
        "markdown fences. It must validate against this JSON Schema:\n\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}\n\n"
        "Every required field must be present. For enum fields use exactly one "
        "of the listed values."
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Models wrap JSON in ```json fences, prepend "Here is the analysis:", or both.
    A bare ``json.loads`` fails on all of those, so this scans for the first
    balanced ``{...}`` region and parses that. Raising :class:`LLMError` on
    failure keeps the messy reality of model output inside this one function.
    """
    import json

    if not text or not text.strip():
        raise LLMError("model returned an empty response")

    candidate = text.strip()
    if candidate.startswith("```"):
        # Strip a fenced block, keeping the inside.
        lines = candidate.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Scan for the first balanced object, respecting string literals so a brace
    # inside a string does not unbalance the count.
    start = candidate.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(candidate)):
            ch = candidate[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(candidate[start : idx + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = candidate.find("{", start + 1)

    raise LLMError(f"no JSON object found in model response: {text[:200]!r}")


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token).

    Only used by the offline stub, where the alternative is reporting zero token
    usage and making the cost-tracking path untestable.
    """
    return max(1, len(text) // 4)
