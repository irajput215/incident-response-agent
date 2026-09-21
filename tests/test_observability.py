"""Observability: structured logs, and a metrics registry that tells the truth.

The assertions that matter are the awkward ones — that a *failed* operation is
still timed, that two registries are genuinely independent, and that latency
memory is bounded. Each corresponds to a way observability quietly stops being
observability.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC

import pytest

from app.observability import Metrics, get_logger, log, setup_logging


# --------------------------------------------------------------------------- #
# structured logging
# --------------------------------------------------------------------------- #
def test_logs_are_one_json_object_per_line(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO", json_output=True)
    log(get_logger("test"), logging.INFO, "incident_triaged", incident_id="INC-1", severity="HIGH")

    line = capsys.readouterr().err.strip().splitlines()[-1]
    payload = json.loads(line)  # must be parseable, not just "looks like JSON"
    assert payload["msg"] == "incident_triaged"
    assert payload["level"] == "INFO"
    assert payload["incident_id"] == "INC-1"
    assert payload["severity"] == "HIGH"
    assert "ts" in payload


def test_the_event_name_is_machine_readable_not_a_sentence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`msg` has to be queryable: `msg="tool_called"`, never prose."""
    setup_logging("INFO", json_output=True)
    log(get_logger("test"), logging.INFO, "tool_called", tool="run_sql")
    payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert " " not in payload["msg"]
    assert payload["msg"] == "tool_called"


def test_complex_field_values_are_coerced_not_crashed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A log call must never raise because a value was not JSON-native."""
    from datetime import datetime

    setup_logging("INFO", json_output=True)
    log(
        get_logger("test"),
        logging.INFO,
        "weird",
        when=datetime.now(tz=UTC),
        nested={"a": {1, 2}},
        items=[1, "two"],
    )
    payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert isinstance(payload["when"], str)
    assert isinstance(payload["items"], list)


def test_setup_logging_is_idempotent(capsys: pytest.CaptureFixture[str]) -> None:
    """An app factory called repeatedly must not duplicate every line."""
    setup_logging("INFO", json_output=True)
    setup_logging("INFO", json_output=True)
    log(get_logger("test"), logging.INFO, "once")
    lines = [ln for ln in capsys.readouterr().err.strip().splitlines() if '"once"' in ln]
    assert len(lines) == 1


def test_logs_go_to_stderr_so_stdout_stays_clean(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`myplatform eval > report.json` must not get log lines in the file."""
    setup_logging("INFO", json_output=True)
    log(get_logger("test"), logging.INFO, "some_event")
    captured = capsys.readouterr()
    assert "some_event" in captured.err
    assert "some_event" not in captured.out


def test_console_format_is_readable(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO", json_output=False)
    log(get_logger("test"), logging.INFO, "plain_event", k="v")
    err = capsys.readouterr().err
    assert "plain_event" in err and "k=v" in err


def test_exception_details_are_captured(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO", json_output=True)
    try:
        raise ValueError("kaboom")
    except ValueError:
        get_logger("test").exception("node_failed")
    payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert payload["msg"] == "node_failed"
    assert payload["exc_type"] == "ValueError"
    assert "kaboom" in payload["exc_message"]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_a_fresh_registry_is_empty(metrics: Metrics) -> None:
    """Possible only because the registry is injected, not a module global."""
    assert metrics.snapshot()["counters"] == {}
    assert metrics.snapshot()["latency"] == {}


def test_two_registries_do_not_share_state() -> None:
    first, second = Metrics(), Metrics()
    first.incr("x", 3)
    assert first.count("x") == 3
    assert second.count("x") == 0
    assert second.snapshot()["counters"] == {}


def test_a_failed_operation_is_still_timed(metrics: Metrics) -> None:
    """The whole reason the timer uses try/finally instead of yield-then-observe.

    A failing operation is exactly when its latency matters, and the naive
    implementation records nothing for it.
    """
    with pytest.raises(ValueError, match="boom"), metrics.timer("op"):
        raise ValueError("boom")

    latency = metrics.snapshot()["latency"]["op"]
    assert latency["count"] == 1
    assert latency["p50_ms"] >= 0


def test_a_successful_operation_is_timed_too(metrics: Metrics) -> None:
    with metrics.timer("op"):
        pass
    assert metrics.snapshot()["latency"]["op"]["count"] == 1
    assert metrics.count("op.calls") == 1


def test_percentiles_separate_once_there_are_enough_samples(metrics: Metrics) -> None:
    for value in range(1, 101):
        metrics.observe("k", float(value))
    latency = metrics.snapshot()["latency"]["k"]
    assert latency["count"] == 100
    assert latency["p50_ms"] < latency["p95_ms"]
    assert latency["max_ms"] == 100.0


def test_percentiles_are_documented_as_coarse_on_tiny_samples() -> None:
    """p95 of three samples is just the maximum — reported with `count` so a
    reader can tell whether the number means anything yet."""
    metrics = Metrics()
    for value in (10.0, 20.0, 30.0):
        metrics.observe("k", value)
    latency = metrics.snapshot()["latency"]["k"]
    assert latency["p95_ms"] == 30.0
    assert latency["count"] == 3


def test_latency_memory_is_bounded() -> None:
    """Unbounded sample lists are the most likely first production memory problem."""
    metrics = Metrics(max_samples=10)
    for value in range(500):
        metrics.observe("k", float(value))

    latency = metrics.snapshot()["latency"]["k"]
    assert latency["count"] == 10  # kept the most recent
    assert latency["max_ms"] == 499.0


def test_counters_are_thread_safe() -> None:
    """`counters[k] = counters.get(k, 0) + n` is a read-modify-write race."""
    import threading

    metrics = Metrics()

    def bump() -> None:
        for _ in range(200):
            metrics.incr("concurrent")

    threads = [threading.Thread(target=bump) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert metrics.count("concurrent") == 1600


def test_reset_clears_everything(metrics: Metrics) -> None:
    metrics.incr("a")
    with metrics.timer("b"):
        pass
    metrics.reset()
    assert metrics.snapshot() == {"counters": {}, "latency": {}}
