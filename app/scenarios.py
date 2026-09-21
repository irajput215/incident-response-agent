"""The scenario catalogue: a small, self-consistent company with five incidents.

This is the project's most load-bearing fixture, because it is used twice:

* as the **demo estate** — the pipelines, run history, logs and warehouse tables
  the agent investigates, and
* as the **evaluation dataset** — each scenario carries its ``expected`` outcome,
  so the same five cases that make the demo work are the cases CI scores.

Building them as one artefact is deliberate. A demo fixture with no expected
outcome cannot catch a regression, and an eval set that does not run in the demo
rots. Here, if you break the agent, both break.

The five incidents mirror real failure modes:

===================================  ==============================  ==========================
id                                   failure                         expected category
===================================  ==============================  ==========================
``missing_partition``                upstream ingestion never ran     UPSTREAM_DEPENDENCY_FAILURE
``schema_change``                    upstream renamed a column        SCHEMA_CHANGE
``db_connection``                    warehouse refused connections    TRANSIENT_INFRASTRUCTURE
``data_quality``                     null-rate assertion tripped      DATA_QUALITY_FAILURE
``spark_oom``                        executor killed for memory       RESOURCE_EXHAUSTION
===================================  ==============================  ==========================
"""
from __future__ import annotations

import json
import random
import zlib
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings

# Everything is anchored to a fixed date so runs, partitions and log timestamps
# are reproducible. A fixture that changes every day cannot be asserted on.
ANCHOR = date(2026, 9, 21)
PARTITION = ANCHOR - timedelta(days=1)  # the partition that failed to arrive
HISTORY_DAYS = 6

ROWS_PER_PARTITION = 200
NULL_RATE_DEFECT = 0.31  # what the data-quality scenario's bad partition looks like
# Derived, so the numbers quoted in the simulated logs are the numbers actually
# in the database. A fixture whose log says "62 of 200" while the table holds a
# random count teaches the agent to distrust evidence — and you to distrust the
# fixture.
QUALITY_NULLS = round(ROWS_PER_PARTITION * NULL_RATE_DEFECT)


@dataclass(frozen=True, slots=True)
class Scenario:
    """One incident: the alert, the world around it, and the right answer."""

    id: str
    title: str
    pipeline: str
    incident: dict[str, Any]
    pipelines: list[dict[str, Any]]
    table: dict[str, Any]
    failing_log: str
    upstream_logs: dict[str, str] = field(default_factory=dict)
    # Evaluation contract. Kept next to the fixture so the two cannot drift.
    expected_category: str = ""
    expected_action: str = ""
    expected_root_cause_terms: tuple[str, ...] = ()
    # Partition overrides: {date: row_count} for this scenario's table.
    partition_rows: dict[str, int] = field(default_factory=dict)
    # Partition overrides for the null-rate defect: {date: {"column": rate}}.
    partition_nulls: dict[str, dict[str, float]] = field(default_factory=dict)


def _iso(day: date, hour: int, minute: int = 0) -> str:
    return datetime.combine(day, time(hour, minute), tzinfo=UTC).isoformat()


def _incident(
    pipeline: str, run_id: str, error: str, *, table: str, day: date = ANCHOR
) -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "run_id": run_id,
        "status": "FAILED",
        "timestamp": _iso(day, 2, 14),
        "error": error,
        "table": table,
    }


# --------------------------------------------------------------------------- #
# 1 — missing partition, caused by a failed upstream pipeline
# --------------------------------------------------------------------------- #
_CLAIMS_ERROR = (
    "SparkException: Job aborted due to stage failure: FileNotFoundError: "
    f"[Errno 2] No such file or directory: s3://acme-lake/claims-raw/dt={PARTITION.isoformat()}/ "
    "(NoSuchKey: The specified key does not exist)"
)

_CLAIMS_FAIL_LOG = f"""2026-09-21 01:04:02 INFO  [claims_daily] task=customer_claims_daily run_id=run_98321 attempt=1
2026-09-21 01:04:02 INFO  [claims_daily] spark-submit --master yarn --deploy-mode cluster claims_job.py
2026-09-21 01:04:05 INFO  [claims_daily] resolving input partition dt={PARTITION.isoformat()}
2026-09-21 01:04:06 INFO  [claims_daily] listing s3://acme-lake/claims-raw/dt={PARTITION.isoformat()}/
2026-09-21 01:04:07 WARN  [claims_daily] path listing returned 0 objects
2026-09-21 01:04:07 ERROR [claims_daily] NoSuchKey: The specified key does not exist: s3://acme-lake/claims-raw/dt={PARTITION.isoformat()}/
2026-09-21 01:04:08 ERROR [claims_daily] org.apache.spark.SparkException: Job aborted due to stage failure
2026-09-21 01:04:08 ERROR [claims_daily]   at org.apache.spark.scheduler.DAGScheduler.failJobAndIndependentStages(DAGScheduler.scala:2672)
2026-09-21 01:04:09 ERROR [claims_daily] run_id=run_98321 status=FAILED duration=7s
"""

_CLAIMS_UPSTREAM_LOG = f"""2026-09-21 00:30:01 INFO  [claims_ingestion] task=claims_ingestion run_id=run_98288 attempt=1
2026-09-21 00:30:04 INFO  [claims_ingestion] extracting from claims_api endpoint=/v2/claims
2026-09-21 00:30:41 WARN  [claims_ingestion] upstream API returned 503 Service Unavailable after 3 attempts
2026-09-21 00:30:41 ERROR [claims_ingestion] HTTPError: 503 Server Error: Service Unavailable for url: https://claims-api.internal/v2/claims
2026-09-21 00:30:42 ERROR [claims_ingestion] no data written to s3://acme-lake/claims-raw/dt={PARTITION.isoformat()}/
2026-09-21 00:30:42 ERROR [claims_ingestion] run_id=run_98288 status=FAILED duration=41s
"""

_MISSING_PARTITION = Scenario(
    id="missing_partition",
    title="Upstream ingestion failed, so the expected partition never arrived",
    pipeline="customer_claims_daily",
    incident=_incident("customer_claims_daily", "run_98321", _CLAIMS_ERROR, table="claims"),
    pipelines=[
        {
            "name": "claims_ingestion",
            "owner": "ingestion-team@acme.example",
            "schedule": "0 0 * * *",
            "criticality": "high",
            "target_table": "claims_raw",
            "description": "Pulls the daily claims extract from the claims API into the landing zone.",
            "upstream": [],
        },
        {
            "name": "customer_claims_daily",
            "owner": "claims-data@acme.example",
            "schedule": "0 1 * * *",
            "criticality": "high",
            "target_table": "claims",
            "description": "Cleans and conforms the daily claims extract into the claims mart.",
            "upstream": ["claims_ingestion"],
        },
    ],
    table={
        "name": "claims",
        "schema_name": "warehouse",
        "description": "Conformed daily claims, one row per claim.",
        "partition_column": "claim_date",
        "expected_rows_per_partition": ROWS_PER_PARTITION,
        "required_columns": ["claim_id", "member_id", "claim_amount"],
        "null_rate_threshold": 0.05,
    },
    failing_log=_CLAIMS_FAIL_LOG,
    upstream_logs={"claims_ingestion": _CLAIMS_UPSTREAM_LOG},
    expected_category="UPSTREAM_DEPENDENCY_FAILURE",
    expected_action="RERUN_UPSTREAM",
    expected_root_cause_terms=("upstream", "partition"),
    partition_rows={PARTITION.isoformat(): 0},
)


# --------------------------------------------------------------------------- #
# 2 — upstream renamed a column
# --------------------------------------------------------------------------- #
_SCHEMA_ERROR = (
    "AnalysisException: cannot resolve 'policy_type' given input columns: "
    "[policy_id, policy_category, effective_date, premium_band]; "
    "line 12 pos 8"
)

_SCHEMA_FAIL_LOG = f"""2026-09-21 03:00:01 INFO  [policy_dim_refresh] task=policy_dim_refresh run_id=run_99104 attempt=1
2026-09-21 03:00:03 INFO  [policy_dim_refresh] source=warehouse.policy_dim_raw partition={PARTITION.isoformat()}
2026-09-21 03:00:05 INFO  [policy_dim_refresh] read 1842 rows from policy_dim_raw
2026-09-21 03:00:06 ERROR [policy_dim_refresh] AnalysisException: cannot resolve 'policy_type' given input columns: [policy_id, policy_category, effective_date, premium_band]
2026-09-21 03:00:06 ERROR [policy_dim_refresh]   the source schema appears to have changed since 2026-09-19
2026-09-21 03:00:07 ERROR [policy_dim_refresh] run_id=run_99104 status=FAILED duration=6s
"""

_SCHEMA_CHANGE = Scenario(
    id="schema_change",
    title="Upstream renamed policy_type to policy_category without notice",
    pipeline="policy_dim_refresh",
    incident=_incident("policy_dim_refresh", "run_99104", _SCHEMA_ERROR, table="policy_dim"),
    pipelines=[
        {
            "name": "policy_feed_ingest",
            "owner": "ingestion-team@acme.example",
            "schedule": "0 2 * * *",
            "criticality": "medium",
            "target_table": "policy_dim_raw",
            "description": "Lands the policy feed verbatim. Owned by the upstream platform team.",
            "upstream": [],
        },
        {
            "name": "policy_dim_refresh",
            "owner": "product-data@acme.example",
            "schedule": "0 3 * * *",
            "criticality": "high",
            "target_table": "policy_dim",
            "description": "Builds the policy dimension used by every downstream premium calculation.",
            "upstream": ["policy_feed_ingest"],
        },
    ],
    table={
        "name": "policy_dim",
        "schema_name": "warehouse",
        "description": "Policy dimension, one row per policy per effective date.",
        "partition_column": "partition_date",
        "expected_rows_per_partition": ROWS_PER_PARTITION,
        "required_columns": ["policy_id", "policy_category"],
        "null_rate_threshold": 0.05,
    },
    failing_log=_SCHEMA_FAIL_LOG,
    expected_category="SCHEMA_CHANGE",
    expected_action="ALERT_OWNER",
    expected_root_cause_terms=("schema", "column"),
)


# --------------------------------------------------------------------------- #
# 3 — the warehouse refused connections
# --------------------------------------------------------------------------- #
_CONN_ERROR = (
    'psycopg.OperationalError: connection to server at "warehouse.internal" (10.4.2.11), '
    "port 5432 failed: Connection refused\n\tIs the server running on that host and accepting TCP/IP connections?"
)

_CONN_FAIL_LOG = """2026-09-21 02:00:01 INFO  [premium_ledger_sync] task=premium_ledger_sync run_id=run_98777 attempt=1
2026-09-21 02:00:02 INFO  [premium_ledger_sync] connecting to postgresql://warehouse.internal:5432/ledger
2026-09-21 02:00:12 WARN  [premium_ledger_sync] connection attempt 1 timed out after 10s
2026-09-21 02:00:32 WARN  [premium_ledger_sync] connection attempt 2 timed out after 10s
2026-09-21 02:00:42 ERROR [premium_ledger_sync] psycopg.OperationalError: connection to server at "warehouse.internal" (10.4.2.11), port 5432 failed: Connection refused
2026-09-21 02:00:42 ERROR [premium_ledger_sync]   Is the server running on that host and accepting TCP/IP connections?
2026-09-21 02:00:43 ERROR [premium_ledger_sync] run_id=run_98777 status=FAILED duration=42s
"""

_DB_CONNECTION = Scenario(
    id="db_connection",
    title="Warehouse refused connections during the sync window",
    pipeline="premium_ledger_sync",
    incident=_incident("premium_ledger_sync", "run_98777", _CONN_ERROR, table="premium_ledger"),
    pipelines=[
        {
            "name": "premium_ledger_sync",
            "owner": "finance-data@acme.example",
            "schedule": "0 2 * * *",
            "criticality": "high",
            "target_table": "premium_ledger",
            "description": "Syncs premium postings into the finance ledger.",
            "upstream": [],
        }
    ],
    table={
        "name": "premium_ledger",
        "schema_name": "warehouse",
        "description": "Premium postings, one row per posting.",
        "partition_column": "partition_date",
        "expected_rows_per_partition": ROWS_PER_PARTITION,
        "required_columns": ["ledger_id", "member_id", "premium_amount"],
        "null_rate_threshold": 0.05,
    },
    failing_log=_CONN_FAIL_LOG,
    expected_category="TRANSIENT_INFRASTRUCTURE",
    expected_action="RERUN_PIPELINE",
    expected_root_cause_terms=("connection", "transient"),
)


# --------------------------------------------------------------------------- #
# 4 — a data-quality assertion rejected the output
# --------------------------------------------------------------------------- #
_QUALITY_ERROR = (
    "dbt test failure: test.null_rate_eligibility_status — FAIL "
    f"null rate {NULL_RATE_DEFECT:.2f} exceeds threshold 0.05 "
    f"({QUALITY_NULLS} of {ROWS_PER_PARTITION} rows) on warehouse.member_eligibility "
    f"partition {PARTITION.isoformat()}"
)

_QUALITY_FAIL_LOG = f"""2026-09-21 04:10:01 INFO  [member_eligibility_daily] task=member_eligibility_daily run_id=run_99512 attempt=1
2026-09-21 04:10:22 INFO  [member_eligibility_daily] built warehouse.member_eligibility partition={PARTITION.isoformat()} rows={ROWS_PER_PARTITION}
2026-09-21 04:10:23 INFO  [member_eligibility_daily] running dbt tests: not_null, unique, accepted_values, null_rate
2026-09-21 04:10:31 WARN  [member_eligibility_daily] test.null_rate_eligibility_status FAIL null_rate={NULL_RATE_DEFECT:.2f} threshold=0.05
2026-09-21 04:10:31 WARN  [member_eligibility_daily] eligibility_status is null for {QUALITY_NULLS} of {ROWS_PER_PARTITION} rows
2026-09-21 04:10:31 ERROR [member_eligibility_daily] dbt test failure: {QUALITY_NULLS} of {ROWS_PER_PARTITION} rows have a null eligibility_status
2026-09-21 04:10:32 ERROR [member_eligibility_daily] run_id=run_99512 status=FAILED duration=31s
"""

_DATA_QUALITY = Scenario(
    id="data_quality",
    title="Null-rate assertion failed on the eligibility mart",
    pipeline="member_eligibility_daily",
    incident=_incident("member_eligibility_daily", "run_99512", _QUALITY_ERROR, table="member_eligibility"),
    pipelines=[
        {
            "name": "member_eligibility_daily",
            "owner": "member-data@acme.example",
            "schedule": "0 4 * * *",
            "criticality": "high",
            "target_table": "member_eligibility",
            "description": "Derives member eligibility for the current period.",
            "upstream": [],
        }
    ],
    table={
        "name": "member_eligibility",
        "schema_name": "warehouse",
        "description": "Member eligibility, one row per member per period.",
        "partition_column": "partition_date",
        "expected_rows_per_partition": ROWS_PER_PARTITION,
        "required_columns": ["member_id", "eligibility_status"],
        "null_rate_threshold": 0.05,
    },
    failing_log=_QUALITY_FAIL_LOG,
    expected_category="DATA_QUALITY_FAILURE",
    expected_action="ESCALATE",
    expected_root_cause_terms=("null", "quality"),
    # The bad partition really has a 31% null rate, so `check_null_rate`
    # measures a genuine defect rather than being told the answer.
    partition_nulls={PARTITION.isoformat(): {"eligibility_status": NULL_RATE_DEFECT}},
)


# --------------------------------------------------------------------------- #
# 5 — the executor was killed for exceeding memory
# --------------------------------------------------------------------------- #
_OOM_ERROR = (
    "java.lang.OutOfMemoryError: Java heap space\n"
    "\tContainer killed by YARN for exceeding memory limits. "
    "12.3 GB of 10 GB physical memory used. Consider boosting spark.yarn.executor.memoryOverhead."
)

_OOM_FAIL_LOG = """2026-09-21 05:00:01 INFO  [claims_aggregation_monthly] task=claims_aggregation_monthly run_id=run_99801 attempt=1
2026-09-21 05:00:03 INFO  [claims_aggregation_monthly] spark.yarn.executor.memory=8g overhead=1g partitions=2400
2026-09-21 05:12:44 WARN  [claims_aggregation_monthly] GC pause 8.4s on executor 7 (exceeds 5s threshold)
2026-09-21 05:18:02 WARN  [claims_aggregation_monthly] shuffle spill to disk: 42.1 GB
2026-09-21 05:23:19 ERROR [claims_aggregation_monthly] java.lang.OutOfMemoryError: Java heap space
2026-09-21 05:23:20 ERROR [claims_aggregation_monthly] Container killed by YARN for exceeding memory limits. 12.3 GB of 10 GB physical memory used
2026-09-21 05:23:21 ERROR [claims_aggregation_monthly] run_id=run_99801 status=FAILED duration=1400s
"""

_SPARK_OOM = Scenario(
    id="spark_oom",
    title="Monthly aggregation exceeded executor memory",
    pipeline="claims_aggregation_monthly",
    incident=_incident("claims_aggregation_monthly", "run_99801", _OOM_ERROR, table="claims_agg"),
    pipelines=[
        {
            "name": "claims_aggregation_monthly",
            "owner": "claims-data@acme.example",
            "schedule": "0 5 1 * *",
            "criticality": "medium",
            "target_table": "claims_agg",
            "description": "Monthly rollup of claims by member. Resource-heavy.",
            "upstream": ["customer_claims_daily"],
        }
    ],
    table={
        "name": "claims_agg",
        "schema_name": "warehouse",
        "description": "Monthly claims aggregation, one row per member per month.",
        "partition_column": "partition_date",
        "expected_rows_per_partition": ROWS_PER_PARTITION,
        "required_columns": ["member_id", "total_amount"],
        "null_rate_threshold": 0.05,
    },
    failing_log=_OOM_FAIL_LOG,
    expected_category="RESOURCE_EXHAUSTION",
    expected_action="RERUN_PIPELINE",
    expected_root_cause_terms=("memory", "executor"),
)


SCENARIOS: tuple[Scenario, ...] = (
    _MISSING_PARTITION,
    _SCHEMA_CHANGE,
    _DB_CONNECTION,
    _DATA_QUALITY,
    _SPARK_OOM,
)

SCENARIOS_BY_ID: dict[str, Scenario] = {s.id: s for s in SCENARIOS}


# --------------------------------------------------------------------------- #
# Generating the world
# --------------------------------------------------------------------------- #
def _history(start_run: int) -> list[dict[str, Any]]:
    """Six days of successful runs, oldest first."""
    runs: list[dict[str, Any]] = []
    for offset in range(HISTORY_DAYS, 0, -1):
        day = ANCHOR - timedelta(days=offset)
        started = _iso(day, 1, 0)
        runs.append(
            {
                "pipeline": "",  # filled in by the caller
                "run_id": f"run_{start_run + offset}",
                "status": "SUCCESS",
                "partition_date": (day - timedelta(days=1)).isoformat(),
                "started_at": started,
                "finished_at": _iso(day, 1, 4),
                "duration_s": 214.0,
            }
        )
    return runs


def build_estate_payload() -> dict[str, Any]:
    """Merge every scenario into one company-wide estate."""
    pipelines: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    seen_pipelines: set[str] = set()
    seen_tables: set[str] = set()

    for index, scenario in enumerate(SCENARIOS):
        for pipeline in scenario.pipelines:
            if pipeline["name"] in seen_pipelines:
                continue
            seen_pipelines.add(pipeline["name"])
            pipelines.append(pipeline)

        # Six successful days of history for the failing pipeline...
        for run in _history(98000 + index * 100):
            run["pipeline"] = scenario.pipeline
            runs.append(run)

        # ...the failing run itself...
        failing = scenario.incident
        runs.append(
            {
                "pipeline": scenario.pipeline,
                "run_id": failing["run_id"],
                "status": "FAILED",
                "partition_date": PARTITION.isoformat(),
                "started_at": _iso(ANCHOR, 1, 4),
                "finished_at": _iso(ANCHOR, 1, 5),
                "error": failing["error"],
                "duration_s": 60.0,
            }
        )

        # ...and, where one exists, the failed upstream run.
        for name in scenario.upstream_logs:
            runs.append(
                {
                    "pipeline": name,
                    "run_id": f"run_{98000 + index * 100 + 50}",
                    "status": "FAILED",
                    "partition_date": PARTITION.isoformat(),
                    "started_at": _iso(ANCHOR, 0, 30),
                    "finished_at": _iso(ANCHOR, 0, 31),
                    "error": "HTTPError: 503 Server Error: Service Unavailable",
                    "duration_s": 41.0,
                }
            )

        if scenario.table["name"] not in seen_tables:
            seen_tables.add(scenario.table["name"])
            tables.append(scenario.table)

    return {"pipelines": pipelines, "runs": runs, "tables": tables}


def write_estate(settings: Settings, payload: dict[str, Any] | None = None) -> Path:
    settings.ensure_directories()
    path = settings.pipeline_estate_file
    path.write_text(json.dumps(payload or build_estate_payload(), indent=2), encoding="utf-8")
    return path


def write_logs(settings: Settings) -> list[Path]:
    """Write one log file per run: successes, the failure, and its upstream."""
    settings.ensure_directories()
    written: list[Path] = []

    for scenario in SCENARIOS:
        directory = settings.pipeline_log_dir / scenario.pipeline
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{scenario.incident['run_id']}.log"
        target.write_text(scenario.failing_log, encoding="utf-8")
        written.append(target)

        for upstream, log_text in scenario.upstream_logs.items():
            upstream_dir = settings.pipeline_log_dir / upstream
            upstream_dir.mkdir(parents=True, exist_ok=True)
            upstream_file = upstream_dir / f"run_{_upstream_run_number(scenario)}.log"
            upstream_file.write_text(log_text, encoding="utf-8")
            written.append(upstream_file)

        # A short, uneventful log for each successful historical run, so
        # `get_previous_runs` + `get_pipeline_logs` can actually compare a good
        # run against a bad one.
        for offset in range(HISTORY_DAYS, 0, -1):
            day = ANCHOR - timedelta(days=offset)
            success = directory / f"run_{_history_run_number(scenario, offset)}.log"
            success.write_text(
                f"{day.isoformat()} 01:00:00 INFO  [{scenario.pipeline}] task={scenario.pipeline} "
                f"run_id={success.stem} attempt=1\n"
                f"{day.isoformat()} 01:00:02 INFO  [{scenario.pipeline}] reading partition "
                f"{(day - timedelta(days=1)).isoformat()}\n"
                f"{day.isoformat()} 01:03:30 INFO  [{scenario.pipeline}] wrote partition "
                f"{(day - timedelta(days=1)).isoformat()} rows={ROWS_PER_PARTITION}\n"
                f"{day.isoformat()} 01:03:34 INFO  [{scenario.pipeline}] run_id={success.stem} "
                f"status=SUCCESS duration=214s\n",
                encoding="utf-8",
            )
            written.append(success)

    return written


def _history_run_number(scenario: Scenario, offset: int) -> int:
    index = SCENARIOS.index(scenario)
    return 98000 + index * 100 + offset


def _upstream_run_number(scenario: Scenario) -> int:
    index = SCENARIOS.index(scenario)
    return 98000 + index * 100 + 50


# --------------------------------------------------------------------------- #
# Seeding the simulated warehouse
# --------------------------------------------------------------------------- #
_TABLE_DDL: dict[str, str] = {
    "claims": (
        "claim_id TEXT, claim_date DATE, member_id TEXT, claim_amount NUMERIC(12,2), "
        "status TEXT, submitted_at TIMESTAMP"
    ),
    "policy_dim": "policy_id TEXT, policy_category TEXT, effective_date DATE, partition_date DATE",
    "premium_ledger": (
        "ledger_id TEXT, member_id TEXT, premium_amount NUMERIC(12,2), partition_date DATE"
    ),
    "member_eligibility": "member_id TEXT, eligibility_status TEXT, partition_date DATE",
    "claims_agg": "month TEXT, member_id TEXT, total_amount NUMERIC(12,2), partition_date DATE",
}


def warehouse_ddl(schema: str = "warehouse") -> list[str]:
    """``CREATE SCHEMA``/``CREATE TABLE`` statements for the simulated warehouse.

    Plain SQL rather than an ORM: this is *the estate being simulated*, and the
    agent queries it with SQL. Keeping the DDL as SQL keeps the simulation
    honest about what the agent will actually be talking to.
    """
    statements = [f'CREATE SCHEMA IF NOT EXISTS "{schema}"']
    for table, columns in _TABLE_DDL.items():
        statements.append(f'CREATE TABLE IF NOT EXISTS "{schema}"."{table}" ({columns})')
    return statements


def _partitions() -> list[date]:
    """Partitions that exist because a *successful* run wrote them.

    The run on day D processes partition D-1, so six successful days of history
    (ANCHOR-6 .. ANCHOR-1) produce partitions ANCHOR-7 .. ANCHOR-2. The failing
    run's own partition (``PARTITION``) is therefore absent by default — which is
    exactly the condition in the missing-partition scenario, and why scenarios
    that *did* write their partition have to declare it explicitly.
    """
    return [ANCHOR - timedelta(days=offset + 1) for offset in range(HISTORY_DAYS, 0, -1)]


def scenario_partitions(scenario: Scenario) -> list[date]:
    """Every partition this scenario's table should hold, oldest first.

    The union of normal history and whatever the scenario explicitly declares —
    a missing partition, or one written before the job failed its assertions.
    """
    days = set(_partitions())
    for key in (*scenario.partition_rows, *scenario.partition_nulls):
        days.add(date.fromisoformat(key))
    return sorted(days)


def partition_row_counts(scenario: Scenario) -> dict[str, int]:
    """How many rows each partition of this scenario's table should hold."""
    counts = {day.isoformat(): ROWS_PER_PARTITION for day in scenario_partitions(scenario)}
    counts.update(scenario.partition_rows)  # e.g. the missing partition -> 0
    return counts


def generate_rows(scenario: Scenario) -> dict[str, list[tuple]]:
    """Deterministically generate the rows for one scenario's table.

    A fixed seed per scenario means the row contents are reproducible across
    runs, so an assertion on a row count or a null rate is stable.
    """
    # crc32, not hash(): Python randomises string hashing per process
    # (PYTHONHASHSEED), so `hash("missing_partition")` differs between runs and
    # the "reproducible" fixture would quietly change its rows on every seed.
    rng = random.Random(zlib.crc32(scenario.id.encode()))
    table = scenario.table["name"]
    rows: list[tuple] = []
    counts = partition_row_counts(scenario)

    for day in scenario_partitions(scenario):
        key = day.isoformat()
        count = counts.get(key, 0)
        null_rate = scenario.partition_nulls.get(key, {}).get("eligibility_status", 0.0)
        # Exact, not probabilistic: a null rate that lands on 61 or 63 depending
        # on the RNG makes the fixture's own log line a subtle lie.
        null_count = round(null_rate * count)
        null_indices = set(rng.sample(range(count), null_count)) if null_count else set()
        for i in range(count):
            if table == "claims":
                rows.append(
                    (
                        f"CLM-{key}-{i:05d}",
                        key,
                        f"MEM-{rng.randint(10000, 99999)}",
                        round(rng.uniform(25.0, 9500.0), 2),
                        rng.choice(["APPROVED", "PENDING", "REJECTED"]),
                        f"{key} 06:00:00",
                    )
                )
            elif table == "policy_dim":
                rows.append(
                    (
                        f"POL-{key}-{i:05d}",
                        rng.choice(["COMPREHENSIVE", "THIRD_PARTY", "FIRE_THEFT"]),
                        key,
                        key,
                    )
                )
            elif table == "premium_ledger":
                rows.append(
                    (
                        f"LDG-{key}-{i:05d}",
                        f"MEM-{rng.randint(10000, 99999)}",
                        round(rng.uniform(10.0, 1200.0), 2),
                        key,
                    )
                )
            elif table == "member_eligibility":
                rows.append(
                    (
                        f"MEM-{10000 + i}",
                        None
                        if i in null_indices
                        else rng.choice(["ELIGIBLE", "INELIGIBLE", "PENDING"]),
                        key,
                    )
                )
            elif table == "claims_agg":
                rows.append(
                    (
                        key[:7],
                        f"MEM-{rng.randint(10000, 99999)}",
                        round(rng.uniform(100.0, 20000.0), 2),
                        key,
                    )
                )
    return {table: rows}


def seed(settings: Settings | None = None) -> dict[str, int | str]:
    """Materialise the whole simulated world.

    Idempotent: tables are replaced and rows re-inserted, so seeding twice
    leaves the same state. Returns a small manifest of what was written.
    """
    settings = settings or get_settings()

    # Imported here so `app.scenarios` stays importable without a database
    # driver, which is what lets the evaluation dataset be read in CI docs
    # tooling and tests that never touch Postgres.
    from app.db.connection import connect
    from app.db.schema import ensure_schema

    estate_path = write_estate(settings)
    log_paths = write_logs(settings)

    row_total = 0
    with connect(settings) as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            for statement in warehouse_ddl():
                cur.execute(statement)
        for scenario in SCENARIOS:
            for table, rows in generate_rows(scenario).items():
                # DELETE + INSERT inside one transaction, rather than TRUNCATE:
                # this keeps the seed re-runnable without needing to own the table.
                with conn.cursor() as cur:
                    cur.execute(f'DELETE FROM "warehouse"."{table}"')
                    if rows:
                        placeholders = ", ".join(["%s"] * len(rows[0]))
                        cur.executemany(
                            f'INSERT INTO "warehouse"."{table}" VALUES ({placeholders})', rows
                        )
                row_total += len(rows)
        conn.commit()

    return {
        "pipelines": len(build_estate_payload()["pipelines"]),
        "scenarios": len(SCENARIOS),
        "log_files": len(log_paths),
        "warehouse_rows": row_total,
        "estate_file": str(estate_path),
    }


def incident_for(scenario_id: str) -> dict[str, Any]:
    return dict(SCENARIOS_BY_ID[scenario_id].incident)


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = argv if argv is not None else sys.argv[1:]
    command = argv[0] if argv else "seed"

    if command == "seed":
        manifest = seed()
        for key, value in manifest.items():
            print(f"{key:16} {value}")
        return 0

    if command == "list":
        for scenario in SCENARIOS:
            print(f"{scenario.id:20} {scenario.pipeline:28} -> {scenario.expected_category}")
        return 0

    if command == "show":
        target = argv[1] if len(argv) > 1 else SCENARIOS[0].id
        print(json.dumps(SCENARIOS_BY_ID[target].incident, indent=2))
        return 0

    print(f"unknown command {command!r}; expected seed | list | show", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
