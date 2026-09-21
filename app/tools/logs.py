"""Log investigation tools: read a run's log, and search across history.

Logs are the first place an on-call engineer looks, so they are the first tools
the agent gets. They are also the most obvious place to introduce a path
traversal bug, so every path built here is validated twice: the name must match
a strict pattern, **and** the resolved path must still be inside the log root.
The second check is redundant today and is exactly the kind of redundancy that
saves you when someone later changes the first one.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.schemas import EvidenceSource
from app.tools.registry import Tool, ToolContext, ToolOutput, object_schema

# Deliberately strict: no dots-dots, no slashes, no null bytes, no unicode
# lookalikes. Pipeline and run identifiers are machine-generated, so accepting
# anything more exotic buys nothing and costs a traversal vulnerability.
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,63}$")

# Lines worth surfacing first. Ordered by severity so the summary leads with the
# worst thing in the file.
_ERROR_MARKERS = ("ERROR", "FATAL", "CRITICAL", "WARN", "Exception", "Traceback")


def _safe_segment(value: str, field: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.match(value):
        raise ValueError(
            f"{field} must match {_NAME_RE.pattern} (got {value!r}); "
            "identifiers are machine-generated and never contain path separators"
        )
    return value


def _log_path(ctx: ToolContext, pipeline: str, run_id: str) -> Path:
    pipeline = _safe_segment(pipeline, "pipeline")
    run_id = _safe_segment(run_id, "run_id")
    root = Path(ctx.settings.pipeline_log_dir).resolve()
    path = (root / pipeline / f"{run_id}.log").resolve()
    # Redundant with _safe_segment on purpose: if the pattern is ever loosened,
    # this still refuses to read outside the log root.
    if not path.is_relative_to(root):
        raise ValueError(f"refusing to read outside the log directory: {path}")
    return path


def _error_lines(lines: list[str], limit: int = 25) -> list[str]:
    """The lines a human would actually read, worst-first."""
    ranked: list[str] = []
    for marker in _ERROR_MARKERS:
        for line in lines:
            if marker in line and line not in ranked:
                ranked.append(line)
    return ranked[:limit]


# --------------------------------------------------------------------------- #
# get_pipeline_logs
# --------------------------------------------------------------------------- #
def _get_pipeline_logs(
    ctx: ToolContext, pipeline: str, run_id: str | None = None, tail_lines: int = 80
) -> ToolOutput:
    # Validate the name *before* the lookup, not after: otherwise an invalid
    # name returns "no runs recorded", which is technically safe but tells the
    # caller the pipeline does not exist when the real problem is the name.
    pipeline = _safe_segment(pipeline, "pipeline")

    if run_id is None:
        runs = ctx.estate.runs_for(pipeline, limit=1)
        if not runs:
            return ToolOutput(
                summary=f"No runs recorded for pipeline {pipeline!r}",
                data={"pipeline": pipeline, "found": False},
            )
        run_id = runs[0].run_id

    path = _log_path(ctx, pipeline, run_id)
    if not path.exists():
        return ToolOutput(
            summary=f"No log file for {pipeline}/{run_id}",
            data={"pipeline": pipeline, "run_id": run_id, "found": False, "path": str(path)},
        )

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-max(1, tail_lines) :]
    errors = _error_lines(lines)

    return ToolOutput(
        summary=(
            f"{pipeline}/{run_id}: {len(lines)} log lines"
            + (f", {len(errors)} error/warn lines" if errors else "")
        ),
        data={
            "pipeline": pipeline,
            "run_id": run_id,
            "found": True,
            "total_lines": len(lines),
            "text": "\n".join(tail),
            "error_lines": errors,
        },
    )


# --------------------------------------------------------------------------- #
# search_logs
# --------------------------------------------------------------------------- #
def _search_logs(
    ctx: ToolContext,
    pattern: str,
    pipeline: str | None = None,
    run_id: str | None = None,
    limit: int = 20,
) -> ToolOutput:
    """Case-insensitive **substring** search.

    Substring, not regex, and that is a deliberate safety choice: a
    model-supplied regex can backtrack catastrophically, and a search tool that
    can hang the investigation is worse than one that cannot express a
    lookahead. Unusual needs are served better by fetching the log and reading it.
    """
    root = Path(ctx.settings.pipeline_log_dir).resolve()
    needle = pattern.lower()
    limit = max(1, min(int(limit), 100))

    if pipeline:
        directories = [root / _safe_segment(pipeline, "pipeline")]
    else:
        directories = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []

    matches: list[dict[str, object]] = []
    files_scanned = 0
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.log")):
            if run_id and path.stem != run_id:
                continue
            files_scanned += 1
            for number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                if needle in line.lower():
                    matches.append(
                        {
                            "pipeline": directory.name,
                            "run_id": path.stem,
                            "line_number": number,
                            "text": line.strip(),
                        }
                    )
                    if len(matches) >= limit:
                        break
            if len(matches) >= limit:
                break
        if len(matches) >= limit:
            break

    scope = pipeline or "all pipelines"
    return ToolOutput(
        summary=(
            f"search {pattern!r} in {scope}: {len(matches)} match(es) "
            f"across {files_scanned} log file(s)"
        ),
        data={
            "pattern": pattern,
            "scope": scope,
            "matches": matches,
            "files_scanned": files_scanned,
            "truncated": len(matches) >= limit,
        },
    )


def build_tools(ctx: ToolContext) -> list[Tool]:
    """The log-investigation tools, with ``ctx`` already bound."""
    return [
        Tool(
            name="get_pipeline_logs",
            description=(
                "Fetch the log of a pipeline run. Returns the tail of the log plus "
                "the error and warning lines. Omit run_id to use the most recent run."
            ),
            parameters=object_schema(
                {
                    "pipeline": {"type": "string", "description": "Pipeline name"},
                    "run_id": {
                        "type": "string",
                        "description": "Specific run to fetch; defaults to the latest run",
                    },
                    "tail_lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 2000,
                        "description": "How many trailing lines to return (default 80)",
                    },
                },
                required=["pipeline"],
            ),
            fn=lambda **kwargs: _get_pipeline_logs(ctx, **kwargs),
            source=EvidenceSource.LOGS,
        ),
        Tool(
            name="search_logs",
            description=(
                "Case-insensitive substring search across pipeline logs. Use it to "
                "find a specific error in the current run, or to check whether the "
                "same error appeared in earlier runs."
            ),
            parameters=object_schema(
                {
                    "pattern": {"type": "string", "description": "Literal text to find"},
                    "pipeline": {
                        "type": "string",
                        "description": "Restrict to one pipeline; omit to search all",
                    },
                    "run_id": {"type": "string", "description": "Restrict to one run"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum matches to return (default 20)",
                    },
                },
                required=["pattern"],
            ),
            fn=lambda **kwargs: _search_logs(ctx, **kwargs),
            source=EvidenceSource.LOGS,
        ),
    ]
