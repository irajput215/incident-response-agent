"""Pipeline-metadata tools: who owns this, what does it depend on, what changed.

These are cheap tools with a high payoff. The single most useful fact in an
upstream-failure investigation is *"this pipeline depends on one that also
failed"* — and that fact lives in metadata, not in the logs, which is exactly
why an agent that only greps logs will confidently report the wrong cause.
"""
from __future__ import annotations

from app.schemas import EvidenceSource
from app.tools.registry import Tool, ToolContext, ToolOutput, object_schema


def _get_pipeline_metadata(ctx: ToolContext, pipeline: str) -> ToolOutput:
    described = ctx.estate.describe(pipeline)
    if not described.get("known"):
        known = ", ".join(sorted(ctx.estate.pipelines)) or "none"
        return ToolOutput(
            summary=f"Pipeline {pipeline!r} is not in the estate metadata; known pipelines: {known}",
            data={"pipeline": pipeline, "known": False, "known_pipelines": sorted(ctx.estate.pipelines)},
        )

    upstream = described["upstream"]
    downstream = described["downstream"]
    return ToolOutput(
        summary=(
            f"{pipeline}: owner={described['owner']}, criticality={described['criticality']}, "
            f"upstream={upstream or 'none'}, downstream={downstream or 'none'}"
        ),
        data=described,
    )


def _get_previous_runs(ctx: ToolContext, pipeline: str, limit: int = 5) -> ToolOutput:
    limit = max(1, min(int(limit), 50))
    runs = ctx.estate.runs_for(pipeline, limit=limit)

    if not runs:
        return ToolOutput(
            summary=f"No runs recorded for pipeline {pipeline!r}",
            data={"pipeline": pipeline, "runs": [], "found": False},
        )

    successes = [r for r in runs if r.succeeded]
    failures = [r for r in runs if r.failed]
    latest = runs[0]

    return ToolOutput(
        summary=(
            f"{pipeline}: {len(runs)} recent run(s) — "
            f"{len(successes)} succeeded, {len(failures)} failed; "
            f"latest {latest.run_id} is {latest.status}"
        ),
        data={
            "pipeline": pipeline,
            "found": True,
            # Deliberately includes partition_date: "which partition did the last
            # good run process?" is the question that turns a failure into a
            # missing-input diagnosis.
            "runs": [r.as_dict() for r in runs],
            "last_success": successes[0].as_dict() if successes else None,
            "failure_count": len(failures),
        },
    )


def build_tools(ctx: ToolContext) -> list[Tool]:
    """Pipeline-metadata tools, with ``ctx`` already bound."""
    return [
        Tool(
            name="get_pipeline_metadata",
            description=(
                "Get a pipeline's owner, schedule, criticality, target table, and its "
                "upstream and downstream dependencies. Use it to find what else could "
                "have caused this failure, or is about to break because of it."
            ),
            parameters=object_schema(
                {"pipeline": {"type": "string", "description": "Pipeline name"}},
                required=["pipeline"],
            ),
            fn=lambda **kwargs: _get_pipeline_metadata(ctx, **kwargs),
            source=EvidenceSource.PIPELINE_HISTORY,
        ),
        Tool(
            name="get_previous_runs",
            description=(
                "Get recent run history for a pipeline, including status and the "
                "partition each run processed. Use it to tell a one-off failure from a "
                "recurring one, and to find the last successful run to compare against."
            ),
            parameters=object_schema(
                {
                    "pipeline": {"type": "string", "description": "Pipeline name"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "How many recent runs to return (default 5)",
                    },
                },
                required=["pipeline"],
            ),
            fn=lambda **kwargs: _get_previous_runs(ctx, **kwargs),
            source=EvidenceSource.PIPELINE_HISTORY,
        ),
    ]
