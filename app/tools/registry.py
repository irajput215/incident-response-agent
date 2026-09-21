"""The tool registry: typed, validated, timed, and impossible to crash.

A ``Tool`` is four things — **name, prose description, JSON Schema,
implementation**. The schema is the model's API; that is what makes tool calling
typed rather than stringly. A tool with no schema is a function call waiting to
be handed the wrong argument.

Three rules the registry enforces so no individual tool has to remember them:

1. **Arguments are validated against the schema before the tool runs.** A model
   that invents an argument gets a structured rejection it can correct, not a
   ``TypeError`` from four frames deep.
2. **A tool can never raise into the agent loop.** Every exception becomes an
   ``ok=False`` result. The loop's job is to reason about failures; it cannot do
   that if a failed tool takes the process down.
3. **Every call is timed and counted**, on the failure path too. Tool error rate
   is an evaluation dimension, and it is unmeasurable if failures are not
   recorded.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

import jsonschema

from app.config import Settings
from app.estate import PipelineEstate
from app.llm.base import ToolSpec
from app.observability import METRICS, Metrics, get_logger, log
from app.schemas import EvidenceSource

_log = get_logger("app.tools")


@dataclass(slots=True)
class ToolOutput:
    """What a tool returns: a human-readable finding plus its structured detail."""

    summary: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    """The envelope the agent loop sees. Always present, never raises."""

    ok: bool
    tool: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "summary": self.summary,
            "data": self.data,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass(slots=True)
class Tool:
    """One capability the agent may invoke."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., ToolOutput]
    source: EvidenceSource = EvidenceSource.LOGS

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)


@dataclass
class ToolContext:
    """Everything the tools need, in one object.

    Tools receive this instead of reaching for globals, so a test can build a
    context pointing at a throwaway estate and a throwaway database.
    """

    settings: Settings
    estate: PipelineEstate
    metrics: Metrics = field(default_factory=lambda: METRICS)


class ToolRegistry:
    """Holds the tools and is the only sanctioned way to call one."""

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx
        self._tools: dict[str, Tool] = {}

    # --- registration ------------------------------------------------------
    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool:
        """Look up a tool.

        The ``KeyError`` deliberately lists the known names: this message is fed
        straight back to a model that hallucinated a tool name, and "no such
        tool" alone invites it to hallucinate a different one.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(
                f"unknown tool {name!r}; available tools: {', '.join(self.names())}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec() for name in self.names()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # --- invocation --------------------------------------------------------
    def validate(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Return an error message, or ``None`` when the arguments are valid."""
        tool = self.get(name)
        try:
            jsonschema.validate(instance=arguments, schema=tool.parameters)
        except jsonschema.ValidationError as exc:
            path = "/".join(str(p) for p in exc.absolute_path) or "(root)"
            return f"invalid arguments for {name}: {exc.message} at {path}"
        except jsonschema.SchemaError as exc:  # a bug in our schema, not the model's
            return f"tool {name} has an invalid schema: {exc.message}"
        return None

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Validate and run a tool. Never raises for a tool-level failure.

        An unknown tool name is one of those failures, not an exception. The
        planner is supposed to have already dropped hallucinated tools, but
        defence in depth means the call path has to survive a name that slipped
        through — and "the run died because the model invented a tool" is not an
        acceptable failure mode for an incident-response system.
        """
        arguments = arguments or {}

        if name not in self._tools:
            self._ctx.metrics.incr("tool.unknown")
            message = (
                f"unknown tool {name!r}; available tools: {', '.join(self.names())}"
            )
            log(_log, 30, "tool_unknown", tool=name)
            return ToolResult(ok=False, tool=name, summary=message, error=message)

        error = self.validate(name, arguments)
        if error is not None:
            self._ctx.metrics.incr("tool.invalid_arguments")
            log(_log, 30, "tool_invalid_arguments", tool=name, error=error[:200])
            return ToolResult(ok=False, tool=name, summary=error, error=error)

        tool = self._tools[name]
        started = perf_counter()
        try:
            output = tool.fn(**arguments)
        except Exception as exc:
            elapsed = (perf_counter() - started) * 1000
            # Collapse newlines: database errors arrive multi-line ("... does not
            # exist\nLINE 2: ..."), and a raw newline in a structured log field
            # makes the log line unparseable.
            detail = " ".join(f"{type(exc).__name__}: {exc}".split())
            self._ctx.metrics.observe(f"tool.{name}.latency_ms", elapsed)
            self._ctx.metrics.incr(f"tool.{name}.errors")
            log(
                _log,
                40,
                "tool_failed",
                tool=name,
                error=detail[:300],
                duration_ms=round(elapsed, 2),
            )
            return ToolResult(
                ok=False,
                tool=name,
                summary=f"{name} failed: {detail}",
                error=detail,
                duration_ms=elapsed,
            )

        elapsed = (perf_counter() - started) * 1000
        self._ctx.metrics.observe(f"tool.{name}.latency_ms", elapsed)
        self._ctx.metrics.incr(f"tool.{name}.calls")
        log(
            _log,
            20,
            "tool_called",
            tool=name,
            duration_ms=round(elapsed, 2),
            summary=output.summary[:200],
        )
        return ToolResult(
            ok=True,
            tool=name,
            summary=output.summary,
            data=output.data,
            duration_ms=elapsed,
        )


class ToolFunction(Protocol):
    """Type of a tool implementation. Documented for readers, not enforced."""

    def __call__(self, **kwargs: Any) -> ToolOutput: ...


def object_schema(
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Build a JSON Schema object with ``additionalProperties: false``.

    Forbidding extra properties is what turns "the model passed a plausible but
    wrong key" into a rejection the agent can see, instead of a silently ignored
    argument and a confusing result.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


__all__ = [
    "Tool",
    "ToolContext",
    "ToolOutput",
    "ToolRegistry",
    "ToolResult",
    "object_schema",
]
