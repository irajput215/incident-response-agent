"""The tool registry and the nine tools, including their attack surface.

The contract under test is that **`registry.call()` never raises**. The agent
loop's job is to reason about failures; it cannot do that if a bad tool name or a
malformed argument takes the process down. Several tests here exist purely to
pin that.
"""
from __future__ import annotations

import json

from app.schemas import EvidenceSource


# --------------------------------------------------------------------------- #
# registry contract
# --------------------------------------------------------------------------- #
def test_every_tool_exposes_a_name_description_and_schema(registry) -> None:
    specs = registry.specs()
    assert len(specs) == 9
    for spec in specs:
        assert spec.name
        assert len(spec.description) > 30, f"{spec.name} needs a real description for the model"
        assert spec.parameters["type"] == "object"
        assert spec.parameters.get("additionalProperties") is False


def test_the_expected_tool_set_is_registered(registry) -> None:
    assert set(registry.names()) == {
        "get_pipeline_logs",
        "search_logs",
        "run_sql",
        "check_table_schema",
        "check_row_count",
        "check_null_rate",
        "check_latest_partition",
        "get_pipeline_metadata",
        "get_previous_runs",
    }


def test_unknown_tool_lookup_lists_the_alternatives(registry) -> None:
    """Fed straight back to a model that hallucinated a name, so it can retry."""
    try:
        registry.get("no_such_tool")
    except KeyError as exc:
        assert "get_pipeline_logs" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")


def test_call_never_raises_for_an_unknown_tool(registry) -> None:
    result = registry.call("no_such_tool", {})
    assert result.ok is False
    assert "unknown tool" in result.summary
    assert "get_pipeline_logs" in result.error


def test_call_never_raises_for_extra_arguments(registry) -> None:
    result = registry.call("check_row_count", {"table": "claims", "bogus": 1})
    assert result.ok is False
    assert "Additional properties" in result.summary


def test_call_never_raises_for_a_missing_required_argument(registry) -> None:
    result = registry.call("check_row_count", {})
    assert result.ok is False
    assert "invalid arguments" in result.summary


def test_call_never_raises_when_the_tool_itself_fails(registry) -> None:
    result = registry.call("check_row_count", {"table": "not_a_table"})
    assert result.ok is False
    assert "unknown table" in result.error


def test_tool_errors_are_single_line(registry) -> None:
    """Database errors arrive multi-line; a newline in a structured log field
    makes the log line unparseable."""
    result = registry.call("run_sql", {"query": "SELECT * FROM warehouse.nope"})
    assert result.ok is False
    assert "\n" not in (result.error or "")


def test_failures_are_counted_in_metrics(registry, metrics) -> None:
    registry.call("nonexistent")
    registry.call("check_row_count", {"table": "nope"})
    counters = metrics.snapshot()["counters"]
    assert counters["tool.unknown"] == 1
    assert counters.get("tool.check_row_count.errors", 0) >= 1


def test_successful_calls_are_timed(registry, metrics) -> None:
    registry.call("check_latest_partition", {"table": "claims"})
    latency = metrics.snapshot()["latency"]
    assert "tool.check_latest_partition.latency_ms" in latency
    assert latency["tool.check_latest_partition.latency_ms"]["count"] == 1


# --------------------------------------------------------------------------- #
# the read path
# --------------------------------------------------------------------------- #
def test_a_write_through_the_read_path_is_rejected_and_changes_nothing(registry) -> None:
    before = registry.call("check_row_count", {"table": "claims", "partition": "2026-09-19"})
    assert before.ok and before.data["row_count"] == 200

    attack = registry.call("run_sql", {"query": "DROP TABLE warehouse.claims"})
    assert attack.data["rejected"] is True
    assert "read-only" in attack.summary

    after = registry.call("check_row_count", {"table": "claims", "partition": "2026-09-19"})
    assert after.ok and after.data["row_count"] == 200, "the table must still be there"


def test_the_multi_statement_bypass_is_rejected(registry) -> None:
    result = registry.call("run_sql", {"query": "select 1; drop table warehouse.claims"})
    assert result.data["rejected"] is True
    assert "one statement" in result.data["reason"]


def test_rejection_is_recorded_as_a_tool_outcome_not_a_crash(registry, metrics) -> None:
    """`ok=True` with `rejected=True`: the tool worked, the query was refused.

    That distinction matters for evaluation — "did the agent attempt a write?" is
    a safety metric, and it is invisible if a rejection looks like a tool error.
    """
    result = registry.call("run_sql", {"query": "INSERT INTO warehouse.claims VALUES ('x')"})
    assert result.ok is True
    assert result.data["rejected"] is True
    assert metrics.snapshot()["counters"]["tool.run_sql.rejected"] == 1


def test_a_real_read_query_returns_rows(registry) -> None:
    result = registry.call(
        "run_sql",
        {
            "query": (
                "SELECT claim_date::text AS d, count(*) AS n FROM warehouse.claims "
                "GROUP BY 1 ORDER BY 1 DESC LIMIT 3"
            )
        },
    )
    assert result.ok is True
    assert result.data["row_count"] == 3
    assert all("d" in row and "n" in row for row in result.data["rows"])


def test_query_results_are_json_safe(registry) -> None:
    """Decimal and date objects would break json.dumps and the trace payload."""
    result = registry.call(
        "run_sql", {"query": "SELECT claim_amount, submitted_at FROM warehouse.claims LIMIT 1"}
    )
    assert result.ok is True
    json.dumps(result.to_dict())  # must not raise


# --------------------------------------------------------------------------- #
# log tools
# --------------------------------------------------------------------------- #
def test_logs_are_returned_with_the_error_lines_surfaced(registry) -> None:
    result = registry.call(
        "get_pipeline_logs", {"pipeline": "customer_claims_daily", "run_id": "run_98321"}
    )
    assert result.ok is True
    assert result.data["found"] is True
    assert result.data["error_lines"], "error lines must be surfaced for the agent"
    assert any("NoSuchKey" in line for line in result.data["error_lines"])


def test_logs_default_to_the_latest_run(registry) -> None:
    result = registry.call("get_pipeline_logs", {"pipeline": "customer_claims_daily"})
    assert result.ok is True
    assert result.data["run_id"] == "run_98321"


def test_a_missing_log_is_reported_not_raised(registry) -> None:
    result = registry.call(
        "get_pipeline_logs", {"pipeline": "customer_claims_daily", "run_id": "run_does_not_exist"}
    )
    assert result.ok is True
    assert result.data["found"] is False


def test_path_traversal_is_refused(registry) -> None:
    """Validated twice: a strict name pattern *and* a resolved-path containment check."""
    for evil in ("../../etc/passwd", "..", "a/b", "./x"):
        result = registry.call("get_pipeline_logs", {"pipeline": evil})
        assert result.ok is False, f"{evil!r} should be refused"
    result = registry.call(
        "get_pipeline_logs", {"pipeline": "customer_claims_daily", "run_id": "../../secrets"}
    )
    assert result.ok is False


def test_search_is_a_literal_substring_not_a_regex(registry) -> None:
    """A model-supplied regex can backtrack catastrophically; a hang is worse
    than an inexpressible query."""
    result = registry.call("search_logs", {"pattern": "NoSuchKey", "pipeline": "customer_claims_daily"})
    assert result.ok is True
    assert result.data["matches"]
    assert all("NoSuchKey" in m["text"] for m in result.data["matches"])

    # A regex metacharacter is treated literally and simply finds nothing.
    result = registry.call("search_logs", {"pattern": "N.*Key", "pipeline": "customer_claims_daily"})
    assert result.ok is True
    assert result.data["matches"] == []


# --------------------------------------------------------------------------- #
# warehouse tools
# --------------------------------------------------------------------------- #
def test_row_count_flags_the_empty_partition(registry) -> None:
    result = registry.call("check_row_count", {"table": "claims", "partition": "2026-09-20"})
    assert result.ok is True
    assert result.data["row_count"] == 0
    assert result.data["is_empty"] is True
    assert result.data["previous_row_count"] == 200
    assert "EMPTY" in result.summary


def test_latest_partition_detects_that_the_expected_one_never_arrived(registry) -> None:
    result = registry.call("check_latest_partition", {"table": "claims"})
    assert result.ok is True
    assert result.data["latest_partition"] == "2026-09-19"


def test_null_rate_measures_the_real_defect(registry) -> None:
    """The fixture's bad partition genuinely has 62 nulls out of 200."""
    result = registry.call(
        "check_null_rate",
        {
            "table": "member_eligibility",
            "column": "eligibility_status",
            "partition": "2026-09-20",
        },
    )
    assert result.ok is True
    assert result.data["null_count"] == 62
    assert result.data["total_rows"] == 200
    assert result.data["null_rate"] == 0.31
    assert result.data["breaches_threshold"] is True


def test_a_healthy_partition_does_not_breach(registry) -> None:
    result = registry.call(
        "check_null_rate",
        {
            "table": "member_eligibility",
            "column": "eligibility_status",
            "partition": "2026-09-19",
        },
    )
    assert result.ok is True
    assert result.data["breaches_threshold"] is False


def test_an_unknown_column_lists_the_real_ones(registry) -> None:
    result = registry.call(
        "check_null_rate", {"table": "member_eligibility", "column": "nope"}
    )
    assert result.ok is False
    assert "eligibility_status" in result.error


def test_schema_tool_reports_required_columns(registry) -> None:
    result = registry.call("check_table_schema", {"table": "policy_dim"})
    assert result.ok is True
    assert result.data["exists"] is True
    names = {c["column_name"] for c in result.data["columns"]}
    assert {"policy_id", "policy_category"} <= names
    assert result.data["missing_required_columns"] == []


def test_schema_tool_detects_a_missing_column(registry) -> None:
    """The schema-change scenario's finding, reproduced directly."""
    result = registry.call("check_table_schema", {"table": "policy_dim"})
    columns = {c["column_name"] for c in result.data["columns"]}
    assert "policy_type" not in columns, "the renamed column is the whole point"
    assert "policy_category" in columns


# --------------------------------------------------------------------------- #
# pipeline metadata tools
# --------------------------------------------------------------------------- #
def test_metadata_exposes_the_dependency_graph(registry) -> None:
    result = registry.call("get_pipeline_metadata", {"pipeline": "customer_claims_daily"})
    assert result.ok is True
    assert result.data["upstream"] == ["claims_ingestion"]
    assert result.data["owner"]


def test_metadata_reports_downstream_dependents(registry) -> None:
    """The direction that answers "what else is about to break?"."""
    result = registry.call("get_pipeline_metadata", {"pipeline": "customer_claims_daily"})
    assert "claims_aggregation_monthly" in result.data["downstream"]


def test_history_separates_successes_from_failures(registry) -> None:
    result = registry.call("get_previous_runs", {"pipeline": "customer_claims_daily", "limit": 10})
    assert result.ok is True
    assert result.data["failure_count"] >= 1
    assert result.data["last_success"] is not None


def test_history_of_the_upstream_pipeline_shows_its_failure(registry) -> None:
    """This single fact is what turns the incident from a data-source problem
    into an upstream-dependency problem."""
    result = registry.call("get_previous_runs", {"pipeline": "claims_ingestion", "limit": 5})
    assert result.ok is True
    assert result.data["failure_count"] == 1


def test_unknown_pipeline_is_reported_not_raised(registry) -> None:
    result = registry.call("get_pipeline_metadata", {"pipeline": "not_a_pipeline"})
    assert result.ok is True
    assert result.data["known"] is False


def test_evidence_source_is_declared_per_tool(registry) -> None:
    assert registry.get("run_sql").source is EvidenceSource.SQL
    assert registry.get("get_pipeline_logs").source is EvidenceSource.LOGS
    assert registry.get("get_pipeline_metadata").source is EvidenceSource.PIPELINE_HISTORY
    assert registry.get("check_table_schema").source is EvidenceSource.SCHEMA
