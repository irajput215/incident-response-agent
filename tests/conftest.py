"""Shared test fixtures.

Two rules keep this suite fast and hermetic:

**No `.env` leaking in.** Every ``Settings`` here is built with
``_env_file=None``, so a developer's real API key cannot change what CI tests.
A suite whose behaviour depends on the machine it runs on is not a suite.

**Integration tests use a throwaway database**, created once per session and
truncated between tests, so a failing test cannot corrupt the demo data and the
demo cannot make a test pass. Unit tests, which are the majority, need no
database at all — that split is what lets the fast tests stay fast.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from app.config import Settings
from app.estate import PipelineEstate
from app.observability import Metrics

TEST_DB = "ai_incidents_test"
ADMIN_DSN = "postgresql://localhost:5432/postgres"


# --------------------------------------------------------------------------- #
# Database provisioning
# --------------------------------------------------------------------------- #
def _postgres_reachable() -> bool:
    try:
        with psycopg.connect(ADMIN_DSN, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except psycopg.Error:
        return False


POSTGRES_AVAILABLE = _postgres_reachable()

requires_postgres = pytest.mark.skipif(
    not POSTGRES_AVAILABLE, reason="integration test needs a local PostgreSQL"
)


@pytest.fixture(scope="session")
def test_dsn() -> str:
    """Create (once) and return the DSN of the throwaway test database."""
    if not POSTGRES_AVAILABLE:
        pytest.skip("PostgreSQL is not reachable")
    with psycopg.connect(ADMIN_DSN, autocommit=True, connect_timeout=3) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", [TEST_DB]
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    return f"postgresql://localhost:5432/{TEST_DB}"


@pytest.fixture(scope="session")
def test_settings(test_dsn: str, tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Settings pointing at throwaway storage and the throwaway database."""
    base: Path = tmp_path_factory.mktemp("estate")
    return Settings(
        _env_file=None,  # never read a developer's .env
        database_url=test_dsn,
        pipeline_estate_file=base / "estate.json",
        pipeline_log_dir=base / "logs",
        llm_model="",  # the deterministic offline analyst
        api_max_rows=100,
        log_level="WARNING",
        log_json=False,
    )


@pytest.fixture(scope="session")
def seeded_estate(test_settings: Settings) -> Settings:
    """Materialise the simulated world once per session."""
    from app.scenarios import seed

    seed(test_settings)
    return test_settings


@pytest.fixture
def estate(seeded_estate: Settings) -> PipelineEstate:
    return PipelineEstate.load(seeded_estate)


@pytest.fixture
def conn(seeded_estate: Settings) -> Iterator[psycopg.Connection]:
    """A connection to the test database, with platform state cleared per test."""
    connection = psycopg.connect(seeded_estate.database_url, row_factory=dict_row)
    from app.db.schema import ensure_schema

    ensure_schema(connection)
    with connection.cursor() as cur:
        cur.execute(
            "TRUNCATE platform.reports, platform.approvals, platform.tool_calls, "
            "platform.evidence, platform.agent_steps, platform.investigations, "
            "platform.incidents RESTART IDENTITY CASCADE"
        )
    connection.commit()
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def repository(conn: psycopg.Connection):
    from app.db import IncidentRepository

    return IncidentRepository(conn)


@pytest.fixture
def metrics() -> Metrics:
    """A fresh registry per test, so counter assertions cannot leak."""
    return Metrics()


@pytest.fixture
def tool_ctx(seeded_estate: Settings, estate: PipelineEstate, metrics: Metrics):
    from app.tools import build_context

    return build_context(seeded_estate, estate=estate, metrics=metrics)


@pytest.fixture
def registry(tool_ctx: Any):
    from app.tools import build_registry

    return build_registry(tool_ctx)


@pytest.fixture
def llm():
    from app.llm import HeuristicLLM

    return HeuristicLLM()


@pytest.fixture
def agent(seeded_estate: Settings, repository: Any, metrics: Metrics, llm: Any):
    """An IncidentAgent on the deterministic analyst and the test database."""
    from app.agents import IncidentAgent

    return IncidentAgent(
        settings=seeded_estate, repository=repository, metrics=metrics, llm=llm
    )


@pytest.fixture
def platform(seeded_estate: Settings, repository: Any, metrics: Metrics, llm: Any):
    """A Platform wired to the test database, with an in-memory checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver

    from app.platform import Platform

    return Platform(
        settings=seeded_estate,
        llm=llm,
        metrics=metrics,
        repository=repository,
        checkpointer=MemorySaver(),
    )


@pytest.fixture
def client(platform: Any) -> Iterator[Any]:
    """An HTTP client over the real app, built by the real factory."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(platform=platform, settings=platform.settings)
    with TestClient(app) as test_client:
        yield test_client


@contextlib.contextmanager
def nullcontext_(*args: Any, **kwargs: Any) -> Iterator[None]:
    yield None
