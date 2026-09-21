"""PostgreSQL access: one place that knows how to reach the database.

Two access patterns, both here so nothing else imports a driver:

* :func:`connect` — a short-lived connection for a script or a CLI command.
* :func:`get_pool` — a process-wide pool for the API, where per-request connects
  would be a latency tax and an unbounded connection count.

Neither is a singleton by accident: the pool is cached per DSN, so a test can
point at a different database and get a different pool rather than silently
reusing the production one.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.config import Settings, get_settings
from app.observability import get_logger, log

_log = get_logger("app.db")

# psycopg raises on connect; a 5s cap stops a wrong host from hanging startup
# for the OS-level TCP timeout (which is minutes).
CONNECT_TIMEOUT_S = 5

_POOLS: dict[str, Any] = {}


class DatabaseUnavailable(RuntimeError):
    """The database could not be reached. Distinct from a query error."""


def _dsn(settings: Settings) -> str:
    return settings.database_url


@contextmanager
def connect(
    settings: Settings | None = None,
    *,
    row_factory: Any = dict_row,
    autocommit: bool = False,
) -> Iterator[psycopg.Connection[dict[str, Any]]]:
    """A short-lived connection. Commits on clean exit, rolls back on error."""
    settings = settings or get_settings()
    try:
        conn = psycopg.connect(
            _dsn(settings),
            row_factory=row_factory,
            connect_timeout=CONNECT_TIMEOUT_S,
            autocommit=autocommit,
        )
    except psycopg.OperationalError as exc:
        raise DatabaseUnavailable(f"cannot reach {_dsn(settings)}: {exc}") from exc

    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


def get_pool(settings: Settings | None = None) -> Any:
    """A cached connection pool for long-lived processes (the API server)."""
    settings = settings or get_settings()
    key = _dsn(settings)
    pool = _POOLS.get(key)
    if pool is None:
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            conninfo=key,
            min_size=settings.db_pool_min,
            max_size=settings.db_pool_max,
            kwargs={"row_factory": dict_row, "connect_timeout": CONNECT_TIMEOUT_S},
            open=True,
        )
        _POOLS[key] = pool
        log(_log, 20, "db_pool_created", dsn=_safe(key), min=settings.db_pool_min, max=settings.db_pool_max)
    return pool


def close_pools() -> None:
    """Close every cached pool. Called on application shutdown and in tests."""
    for key, pool in list(_POOLS.items()):
        try:
            pool.close()
        finally:
            _POOLS.pop(key, None)


def database_available(settings: Settings | None = None) -> bool:
    """Cheap liveness probe.

    Used by ``/health`` and by the integration tests to skip cleanly on a
    machine without Postgres, rather than failing with a driver traceback.
    """
    try:
        with connect(settings) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except (DatabaseUnavailable, psycopg.Error):
        return False


def _safe(dsn: str) -> str:
    from app.config import _redact_dsn

    return _redact_dsn(dsn)
