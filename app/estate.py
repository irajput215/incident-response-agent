"""The simulated company: pipelines, their run history, and their tables.

An incident-response agent is only as good as the estate it can see. This module
is that estate — a small, *self-consistent* world that the agent investigates
through tools, exactly as it would investigate a real one.

Two design decisions worth stating:

**It is data, not code.** The estate is loaded from ``data/estate.json``, which
:mod:`app.scenarios` generates. Swapping the simulated company for a real one
therefore means pointing the loader at real metadata — not rewriting the agent.

**It is deliberately imperfect.** Prior runs contain noise, some pipelines have
no upstream declared, and one scenario's evidence genuinely does not identify a
single cause. A fixture where every failure has an obvious signature teaches you
nothing about an agent, because the agent never has to say "I don't know".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.observability import get_logger, log

_log = get_logger("app.estate")


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


@dataclass(frozen=True, slots=True)
class Pipeline:
    """One scheduled job in the estate."""

    name: str
    owner: str
    schedule: str
    criticality: str  # low | medium | high
    target_table: str
    description: str = ""
    upstream: tuple[str, ...] = ()

    @property
    def is_critical(self) -> bool:
        return self.criticality.lower() == "high"


@dataclass(frozen=True, slots=True)
class PipelineRun:
    """One execution of a pipeline."""

    pipeline: str
    run_id: str
    status: str  # SUCCESS | FAILED | RUNNING
    partition_date: str
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
    duration_s: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.status.upper() == "SUCCESS"

    @property
    def failed(self) -> bool:
        return self.status.upper() == "FAILED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "run_id": self.run_id,
            "status": self.status,
            "partition_date": self.partition_date,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error": self.error,
            "duration_s": round(self.duration_s, 1),
        }


@dataclass(frozen=True, slots=True)
class TableInfo:
    """A table in the simulated warehouse, plus what "normal" looks like for it."""

    name: str
    schema_name: str
    description: str
    partition_column: str
    expected_rows_per_partition: int
    # Columns that must never be null, and the threshold above which a null rate
    # is considered a defect. This is the contract the data-quality tool checks.
    required_columns: tuple[str, ...] = ()
    null_rate_threshold: float = 0.05

    @property
    def qualified(self) -> str:
        return f"{self.schema_name}.{self.name}"


@dataclass
class PipelineEstate:
    """An in-memory view of the estate, loaded once and queried cheaply."""

    pipelines: dict[str, Pipeline] = field(default_factory=dict)
    runs: dict[str, list[PipelineRun]] = field(default_factory=dict)
    tables: dict[str, TableInfo] = field(default_factory=dict)

    # --- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PipelineEstate:
        pipelines: dict[str, Pipeline] = {}
        for raw in payload.get("pipelines", []):
            pipeline = Pipeline(
                name=raw["name"],
                owner=raw.get("owner", "unknown@example.com"),
                schedule=raw.get("schedule", ""),
                criticality=raw.get("criticality", "medium"),
                target_table=raw.get("target_table", ""),
                description=raw.get("description", ""),
                upstream=tuple(raw.get("upstream", ())),
            )
            pipelines[pipeline.name] = pipeline

        runs: dict[str, list[PipelineRun]] = {}
        for raw in payload.get("runs", []):
            run = PipelineRun(
                pipeline=raw["pipeline"],
                run_id=raw["run_id"],
                status=raw["status"],
                partition_date=raw.get("partition_date", ""),
                started_at=_parse_dt(raw.get("started_at")) or datetime.now(tz=UTC),
                finished_at=_parse_dt(raw.get("finished_at")),
                error=raw.get("error"),
                duration_s=float(raw.get("duration_s", 0.0)),
            )
            runs.setdefault(run.pipeline, []).append(run)
        for entries in runs.values():
            # Newest first: every consumer wants recency, and sorting once here
            # beats remembering to sort at each call site.
            entries.sort(key=lambda r: r.started_at, reverse=True)

        tables: dict[str, TableInfo] = {}
        for raw in payload.get("tables", []):
            table = TableInfo(
                name=raw["name"],
                schema_name=raw.get("schema_name", "warehouse"),
                description=raw.get("description", ""),
                partition_column=raw.get("partition_column", "partition_date"),
                expected_rows_per_partition=int(raw.get("expected_rows_per_partition", 0)),
                required_columns=tuple(raw.get("required_columns", ())),
                null_rate_threshold=float(raw.get("null_rate_threshold", 0.05)),
            )
            tables[table.name] = table

        return cls(pipelines=pipelines, runs=runs, tables=tables)

    @classmethod
    def load(cls, settings: Settings | None = None) -> PipelineEstate:
        """Load the estate, or raise if it has not been seeded.

        Raising is right for the CLI and the tests: both are interactive, and a
        missing estate there is a mistake worth stopping for.
        """
        settings = settings or get_settings()
        path: Path = settings.pipeline_estate_file
        if not path.exists():
            raise FileNotFoundError(
                f"pipeline estate not found at {path}. Run `adp-agent seed` first."
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def load_or_empty(cls, settings: Settings | None = None) -> PipelineEstate:
        """Load the estate, or return an empty one **and say so loudly**.

        Used by the server. A long-running process should start even when its
        estate has not been seeded — refusing to boot turns a missing fixture into
        an outage — but it must not pretend everything is fine either. The warning
        is logged here and the emptiness is reported by ``/health`` as
        ``estate.seeded: false``, so "the agent says it knows no pipelines" has an
        immediate, visible explanation.
        """
        settings = settings or get_settings()
        try:
            return cls.load(settings)
        except FileNotFoundError:
            log(
                _log,
                30,
                "estate_not_seeded",
                path=str(settings.pipeline_estate_file),
                detail=(
                    "the agent will start but cannot investigate anything; "
                    "run `adp-agent seed`"
                ),
            )
            return cls()

    @property
    def seeded(self) -> bool:
        """Whether any pipeline metadata is loaded."""
        return bool(self.pipelines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipelines": [
                {
                    "name": p.name,
                    "owner": p.owner,
                    "schedule": p.schedule,
                    "criticality": p.criticality,
                    "target_table": p.target_table,
                    "description": p.description,
                    "upstream": list(p.upstream),
                }
                for p in self.pipelines.values()
            ],
            "runs": [run.as_dict() for runs in self.runs.values() for run in runs],
            "tables": [
                {
                    "name": t.name,
                    "schema_name": t.schema_name,
                    "description": t.description,
                    "partition_column": t.partition_column,
                    "expected_rows_per_partition": t.expected_rows_per_partition,
                    "required_columns": list(t.required_columns),
                    "null_rate_threshold": t.null_rate_threshold,
                }
                for t in self.tables.values()
            ],
        }

    # --- queries -----------------------------------------------------------
    def pipeline(self, name: str) -> Pipeline | None:
        return self.pipelines.get(name)

    def table(self, name: str) -> TableInfo | None:
        return self.tables.get(name)

    def runs_for(self, pipeline: str, limit: int = 10) -> list[PipelineRun]:
        return self.runs.get(pipeline, [])[:limit]

    def run(self, pipeline: str, run_id: str) -> PipelineRun | None:
        return next((r for r in self.runs.get(pipeline, []) if r.run_id == run_id), None)

    def last_success(self, pipeline: str, *, before: datetime | None = None) -> PipelineRun | None:
        """Most recent successful run, optionally only before a given time.

        ``before`` is what makes "what did the last good run look like?" a
        meaningful question during an incident — comparing against a run that
        happened *after* the failure would be nonsense.
        """
        for run in self.runs.get(pipeline, []):
            if not run.succeeded:
                continue
            if before is not None and run.started_at >= before:
                continue
            return run
        return None

    def upstream_of(self, pipeline: str) -> list[str]:
        entry = self.pipelines.get(pipeline)
        return list(entry.upstream) if entry else []

    def downstream_of(self, pipeline: str) -> list[str]:
        """Pipelines that declare ``pipeline`` as an upstream dependency.

        The inverse of ``upstream_of``, and the direction that answers "what
        else is about to break?" — which is the question an on-call engineer
        actually asks at 2am.
        """
        return [
            p.name for p in self.pipelines.values() if pipeline in p.upstream
        ]

    def describe(self, pipeline: str) -> dict[str, Any]:
        """Everything known about a pipeline, for prompt context."""
        entry = self.pipelines.get(pipeline)
        if entry is None:
            return {"pipeline": pipeline, "known": False}
        return {
            "pipeline": entry.name,
            "known": True,
            "owner": entry.owner,
            "schedule": entry.schedule,
            "criticality": entry.criticality,
            "target_table": entry.target_table,
            "description": entry.description,
            "upstream": list(entry.upstream),
            "downstream": self.downstream_of(pipeline),
            "recent_runs": [r.as_dict() for r in self.runs_for(pipeline, limit=5)],
        }
