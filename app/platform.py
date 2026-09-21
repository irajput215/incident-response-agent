"""The platform container: everything a request needs, assembled once.

One object, built at startup, holding the LLM client, the tool registry, the
database connection, the checkpointer and the agent. Requests borrow the agent
rather than constructing one, which is what keeps the graph compiled once and —
more importantly — keeps the **checkpointer identity stable**. An approval pause
stored by one checkpointer instance cannot be resumed by another, so a
per-request agent would make human-in-the-loop quietly impossible.

⚠️ **Known limitation, stated deliberately.** The server shares a single
database connection across requests. psycopg connections serialise access
internally, so this is *correct* but not concurrent: two simultaneous
investigations queue behind each other. A production deployment would give each
request its own repository from the pool. It is one connection here because the
alternative — a repository per request — would have to be threaded through every
graph node, and that refactor is not worth doing before the concurrency is
actually needed.
"""
from __future__ import annotations

import contextlib
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.agents import IncidentAgent
from app.config import Settings, get_settings
from app.db import IncidentRepository, ensure_schema
from app.llm import LLMClient, build_llm
from app.observability import METRICS, Metrics, get_logger, log
from app.tools import ToolRegistry, build_context, build_registry

_log = get_logger("app.platform")


class Platform:
    """The composition root. Imports nothing that imports it."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        metrics: Metrics | None = None,
        registry: ToolRegistry | None = None,
        repository: IncidentRepository | None = None,
        connection: psycopg.Connection[dict[str, Any]] | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.metrics = metrics if metrics is not None else METRICS

        # --- database -------------------------------------------------------
        self._owns_connection = connection is None
        if repository is not None:
            self.repo = repository
        else:
            # Fail fast: a connection failure here raises and the process does
            # not start. That is deliberate for a service whose entire job is
            # reading the warehouse, and it is why `health()`'s unreachable
            # branch is defence-in-depth rather than a state this path reaches.
            self._conn = connection or psycopg.connect(
                self.settings.database_url, row_factory=dict_row, connect_timeout=5
            )
            if self.settings.db_auto_migrate:
                ensure_schema(self._conn)
            self.repo = IncidentRepository(self._conn)

        # --- model and tools -------------------------------------------------
        self.llm = llm or build_llm(self.settings, metrics=self.metrics)
        self.registry = registry or build_registry(
            build_context(self.settings, metrics=self.metrics)
        )

        # --- checkpointing ---------------------------------------------------
        if checkpointer is not None:
            # Supplied by the caller (tests, or an embedder): never build a second
            # one, because two checkpointers cannot resume each other's pauses.
            self.checkpointer, self._checkpointer_conn = checkpointer, None
        else:
            self.checkpointer, self._checkpointer_conn = _build_checkpointer(self.settings)

        self.agent = IncidentAgent(
            settings=self.settings,
            llm=self.llm,
            registry=self.registry,
            repository=self.repo,
            metrics=self.metrics,
            checkpointer=self.checkpointer,
        )

        log(
            _log,
            20,
            "platform_ready",
            model=self.llm.model,
            tools=len(self.registry),
            checkpointer=type(self.checkpointer).__name__,
        )

    # --- diagnostics --------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """Liveness **and** capability.

        Liveness alone ("ok") tells an operator nothing about whether the thing
        that is up is the thing they configured.
        """
        database: dict[str, Any] = {"reachable": False}
        try:
            from app.db.connection import database_available
            from app.db.schema import describe

            if database_available(self.settings):
                database = {"reachable": True, **describe(self.repo._conn)}
                database["counts"] = self.repo.stats()
        except Exception as exc:  # health must never raise
            database = {"reachable": False, "error": str(exc)[:200]}

        return {
            "status": "ok",
            "version": _version(),
            "capabilities": self.settings.capabilities(),
            "database": database,
            "estate": {
                "seeded": self.registry._ctx.estate.seeded,
                "pipelines": len(self.registry._ctx.estate.pipelines),
                "detail": (
                    "run `adp-agent seed` to populate the simulated estate"
                    if not self.registry._ctx.estate.seeded
                    else None
                ),
            },
            "tools": len(self.registry),
            "checkpointer": type(self.checkpointer).__name__,
        }

    def close(self) -> None:
        # Suppressed rather than logged: this runs on the shutdown path, where
        # there is nobody left to act on a failure and raising would mask the
        # real reason the process is stopping.
        checkpointer_conn = getattr(self, "_checkpointer_conn", None)
        with contextlib.suppress(Exception):
            if checkpointer_conn is not None:
                checkpointer_conn.close()
        conn = getattr(self, "_conn", None)
        with contextlib.suppress(Exception):
            if self._owns_connection and conn is not None:
                conn.close()


def _build_checkpointer(settings: Settings) -> tuple[Any, Any]:
    """Return ``(checkpointer, connection_to_close)``.

    Falls back to an in-memory saver, loudly, when Postgres is unavailable — a
    server that refuses to start because checkpointing is degraded is worse than
    one that starts and says so.
    """
    if settings.checkpoint_backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            conn = psycopg.connect(settings.database_url, autocommit=True, connect_timeout=5)
            saver = PostgresSaver(conn)  # type: ignore[arg-type]
            saver.setup()
            log(_log, 20, "checkpointer_ready", backend="postgres")
            return saver, conn
        except Exception as exc:
            log(
                _log,
                30,
                "checkpointer_fallback",
                backend="memory",
                error=str(exc)[:200],
            )

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver(), None


def _version() -> str:
    from app import __version__

    return __version__


__all__ = ["Platform"]
