"""Application factory and ASGI entry point.

``create_app()`` is a **factory**, never a module-level ``app = FastAPI()``. The
difference is not stylistic: a module-level app would build the platform — and
therefore open a database connection and compile the graph — at *import* time.
That makes the module unimportable in a test that has no database, unimportable
in the CLI, and impossible to run two configurations in one process.

Run it with::

    uvicorn app.main:create_app --factory --reload
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api import router
from app.config import Settings, get_settings
from app.observability import get_logger, log, setup_logging
from app.platform import Platform

_log = get_logger("app.main")

DESCRIPTION = """
An agentic AI system that investigates failed data pipelines.

Report a failed run, and a LangGraph workflow triages it, investigates the logs
and the data with typed tools, establishes a root cause, proposes a remediation,
**waits for a human to approve anything that changes data**, and writes an
incident report.

* Every LLM and tool call is traced, counted and timed.
* The SQL tool is a real read-only boundary, not a `startswith("select")` check.
* With no API key configured, the whole system runs a deterministic offline
  analyst and the evaluation suite still passes.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Close the platform's connections on shutdown.

    Without this, a reload leaves the previous process holding the database
    connection until the OS reaps it, and the next startup contends for it.
    """
    platform: Platform = app.state.platform
    log(
        _log,
        20,
        "api_starting",
        model=platform.llm.model,
        tools=len(platform.registry),
        checkpointer=type(platform.checkpointer).__name__,
    )
    try:
        yield
    finally:
        platform.close()
        log(_log, 20, "api_stopped")


def create_app(
    platform: Platform | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Build the ASGI app.

    Passing a ``platform`` lets a test supply one wired to a throwaway database
    and a stub model, with no environment variables involved.
    """
    settings = settings or get_settings()
    setup_logging(settings.log_level, json_output=settings.log_json)

    app = FastAPI(
        title="AI Data Engineering Incident Response Agent",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.platform = platform if platform is not None else Platform(settings)
    app.include_router(router)
    return app


def main() -> int:
    """Console entry point: ``python -m app.main`` or the ``adp-agent`` script."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,  # our own JSON logging is already configured
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
