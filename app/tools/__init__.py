"""The agent's tools, and the registry that binds them together.

Nine tools across three families, each answering a different question an
investigator asks:

===========================  ==================================================
family                       question
===========================  ==================================================
logs                         What did the job say when it died?
pipeline metadata            What is this, who owns it, what does it depend on?
warehouse                    What does the data actually look like right now?
===========================  ==================================================

``build_context()`` and ``build_registry()`` are the only entry points. Fetching
the estate once and passing it in means the tools never re-read ``estate.json``
per call, and a test can hand them a fixture instead of the real thing.
"""
from __future__ import annotations

from app.config import Settings, get_settings
from app.estate import PipelineEstate
from app.observability import METRICS, Metrics
from app.tools import database, logs, pipeline
from app.tools.registry import (
    Tool,
    ToolContext,
    ToolOutput,
    ToolRegistry,
    ToolResult,
    object_schema,
)
from app.tools.sql_guard import UnsafeQuery, assert_read_only, is_read_only

__all__ = [
    "Tool",
    "ToolContext",
    "ToolOutput",
    "ToolRegistry",
    "ToolResult",
    "UnsafeQuery",
    "assert_read_only",
    "build_context",
    "build_registry",
    "is_read_only",
    "object_schema",
]


def build_context(
    settings: Settings | None = None,
    *,
    estate: PipelineEstate | None = None,
    metrics: Metrics | None = None,
    strict: bool = False,
) -> ToolContext:
    """Assemble everything the tools need, once.

    ``strict=False`` (the default, and what the server uses) tolerates an
    unseeded estate so a long-running process still starts; ``strict=True``
    raises, which is what the CLI and tests want.
    """
    settings = settings or get_settings()
    if estate is None:
        estate = (
            PipelineEstate.load(settings)
            if strict
            else PipelineEstate.load_or_empty(settings)
        )
    return ToolContext(
        settings=settings,
        estate=estate,
        metrics=metrics if metrics is not None else METRICS,
    )


def build_registry(ctx: ToolContext) -> ToolRegistry:
    """Register every tool the agent may call."""
    registry = ToolRegistry(ctx)
    registry.register_all(logs.build_tools(ctx))
    registry.register_all(database.build_tools(ctx))
    registry.register_all(pipeline.build_tools(ctx))
    return registry
