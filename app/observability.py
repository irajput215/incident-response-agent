"""Structured logging and an in-process metrics registry.

Two things every later phase depends on, built first because retrofitting
observability into finished code is miserable:

* **``log(logger, level, "event_name", **fields)``** — the ``msg`` is a *stable,
  machine-readable event name* (``incident_triaged``, ``tool_called``,
  ``node_failed``), never a sentence. You must be able to write
  ``msg="tool_called"`` in a query; ``print("called the tool!")`` is unparseable.

* **``Metrics``** — counters and latency percentiles, passed in explicitly
  rather than held in a module-level singleton. A global registry is
  per-process, unrestartable across tests, and impossible to assert on; an
  injected one makes ``metrics.snapshot()["counters"] == {}`` a valid assertion.

The registry is also **bounded** (see ``MAX_SAMPLES``): raw latency samples are
appended to deques with a cap, because an unbounded list in a long-lived server
is the single most likely first production memory problem.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from typing import Any

# Cap on retained latency samples per key. Percentiles over the most recent N
# observations are what you want operationally anyway — and it makes memory
# use a constant rather than a function of uptime.
MAX_SAMPLES = 2048

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Render a log record as one JSON object per line.

    Extra ``key=value`` fields attached via ``log()`` are promoted to top-level
    keys, so a log query is ``fields.incident_id = '...'`` rather than a regex
    over a message string.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _iso_utc(record.created),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = _coerce(value)
        if record.exc_info:
            payload["exc_type"] = getattr(record.exc_info[0], "__name__", "Exception")
            payload["exc_message"] = str(record.exc_info[1])
        return json.dumps(payload, default=str, separators=(",", ":"))


class ConsoleFormatter(logging.Formatter):
    """Human-readable logs for local development."""

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        suffix = " " + " ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        base = f"{_iso_utc(record.created)} {record.levelname:<8} {record.name} {record.getMessage()}"
        if record.exc_info:
            base += f" | {record.exc_info[0].__name__ if record.exc_info[0] else ''}: {record.exc_info[1]}"
        return base + suffix


def _coerce(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    return str(value)


def _iso_utc(epoch: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(epoch, tz=UTC).isoformat(timespec="milliseconds")


def setup_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure the root logger for this application.

    Idempotent: clears existing handlers first, so calling it twice (an app
    factory invoked in several tests) does not duplicate every line.
    """
    handler = logging.StreamHandler(sys.stderr)  # stderr, so stdout stays clean data
    handler.setFormatter(JsonFormatter() if json_output else ConsoleFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # uvicorn installs its own handlers; let them propagate to ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured event.

    ``event`` is a stable identifier, not prose: ``log(log, INFO, "incident_failed",
    incident_id=..., error=...)``. Keeping it short and machine-readable is what
    makes it queryable later.
    """
    logger.log(level, event, extra=fields)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile.

    ⚠️ Known weakness: with few samples this is coarse — p95 of 3 observations is
    just the maximum. It is reported alongside ``count`` so a reader can judge
    whether the number means anything yet. (A t-digest would be the upgrade.)
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return round(sorted_values[0], 3)
    rank = max(1, min(len(sorted_values), int(-(-pct * len(sorted_values) // 100))))
    return round(sorted_values[rank - 1], 3)


class Metrics:
    """Counters and latency distributions. Thread-safe, bounded, injectable.

    Every mutation takes the lock because ``counters[k] = counters.get(k, 0) + n``
    is a read-modify-write race, not an atomic operation.
    """

    def __init__(self, max_samples: int = MAX_SAMPLES) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._latencies: dict[str, deque[float]] = {}
        self._max_samples = max_samples

    # --- counters ----------------------------------------------------------
    def incr(self, key: str, n: float = 1) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + n

    def count(self, key: str) -> float:
        with self._lock:
            return self._counters.get(key, 0)

    # --- latency -----------------------------------------------------------
    def observe(self, key: str, ms: float) -> None:
        with self._lock:
            samples = self._latencies.get(key)
            if samples is None:
                samples = deque(maxlen=self._max_samples)
                self._latencies[key] = samples
            samples.append(ms)

    @contextmanager
    def timer(self, key: str, **fields: Any) -> Iterator[None]:
        """Time a block, recording **on the failure path too**.

        The ``try/finally`` is the whole point: a failing operation is exactly
        when you most want its latency. ``yield`` followed by an ``observe()``
        call would skip every exception.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.observe(key, elapsed_ms)
            self.incr(f"{key}.calls")

    # --- read-out ----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            latency: dict[str, dict[str, float]] = {}
            for key, samples in self._latencies.items():
                ordered = sorted(samples)
                latency[key] = {
                    "count": len(ordered),
                    "p50_ms": _percentile(ordered, 50),
                    "p95_ms": _percentile(ordered, 95),
                    "max_ms": round(ordered[-1], 3) if ordered else 0.0,
                }
        return {"counters": counters, "latency": latency}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latencies.clear()


# A module-level default for convenience — components still receive one by
# injection, so tests can pass their own and get an independent snapshot.
METRICS = Metrics()
