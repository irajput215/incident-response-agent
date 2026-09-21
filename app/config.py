"""Typed runtime configuration — the single source of truth for what this
process believes it has available.

Two design rules, both deliberate:

1. **Capability is derived, never assumed.** ``settings.capabilities()`` answers
   "can this process talk to an LLM / LangSmith / S3?" by inspecting the config
   once, so no node, tool, or route ever has to guess or rummage in ``os.environ``.
   ``/health`` and the logs both report it, which makes a misconfigured
   deployment obvious instead of mysterious.

2. **Every credential is optional.** An empty ``.env`` yields a process that
   runs the deterministic offline stub. That is not a fallback bolted on at the
   end — it is the default, and it is what lets the whole test suite and the
   agent evaluation suite run in CI with no keys and no network.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "ci", "staging", "production"]

# The stub is a real, first-class backend: it is what CI uses, and it makes the
# evaluation suite hermetic. Anything else is a LiteLLM model string.
STUB_MODEL = "stub"


class Settings(BaseSettings):
    """All runtime configuration, read from the environment or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- application -------------------------------------------------------
    app_env: Environment = Field(default="local")
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=True)  # JSON to stderr; False = human console

    # --- LLM (provider-agnostic, via LiteLLM) ------------------------------
    # Empty == run the deterministic stub. See app/llm/stub.py.
    llm_model: str = Field(default="")
    llm_temperature: float = Field(default=0.0)
    llm_timeout_s: float = Field(default=60.0)
    llm_max_retries: int = Field(default=2)
    # A per-incident ceiling on LLM calls. An agent that loops is an agent that
    # spends money; this makes the bound explicit and testable.
    llm_max_calls_per_incident: int = Field(default=25)

    anthropic_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    deepseek_api_key: str = Field(default="")
    ollama_api_base: str = Field(default="http://localhost:11434")

    # --- LangSmith: tracing + evaluation -----------------------------------
    langsmith_tracing: str = Field(default="")
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="ai-incident-agent")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")

    # --- PostgreSQL --------------------------------------------------------
    database_url: str = Field(default="postgresql://localhost:5432/ai_incidents")
    db_auto_migrate: bool = Field(default=True)
    db_pool_min: int = Field(default=1)
    db_pool_max: int = Field(default=8)
    # Where LangGraph stores suspended state. `postgres` is what lets an
    # approval pause survive a server restart — with `memory`, a deploy during an
    # incident silently discards the investigation waiting for a human.
    checkpoint_backend: Literal["memory", "postgres"] = Field(default="postgres")

    # --- simulated pipeline estate -----------------------------------------
    pipeline_log_dir: Path = Field(default=Path("data/logs"))
    pipeline_estate_file: Path = Field(default=Path("data/estate.json"))
    s3_bucket: str = Field(default="")
    aws_region: str = Field(default="ap-southeast-2")
    aws_access_key_id: str = Field(default="")
    aws_secret_access_key: str = Field(default="")

    # --- serving -----------------------------------------------------------
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)
    api_max_rows: int = Field(default=1000)

    # --- agent behaviour ---------------------------------------------------
    max_investigation_rounds: int = Field(default=3)
    # Below this, the root-cause node asks for more evidence instead of
    # committing to an answer.
    root_cause_confidence_threshold: float = Field(default=0.6)

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper().strip()
        if v not in allowed:
            # Fail loudly at startup. A typo'd log level silently disabling
            # your logs is exactly the kind of thing you find out at 2am.
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("db_pool_max")
    @classmethod
    def _pool_sane(cls, v: int) -> int:
        if v < 1:
            raise ValueError("db_pool_max must be >= 1")
        return v

    # --- derived capability -------------------------------------------------
    @property
    def llm_model_id(self) -> str:
        """The effective model id: the configured one, or ``"stub"``."""
        return self.llm_model.strip() or STUB_MODEL

    @property
    def llm_enabled(self) -> bool:
        """True when a real model is configured (i.e. not the stub)."""
        return self.llm_model_id != STUB_MODEL

    @property
    def langsmith_enabled(self) -> bool:
        return self.langsmith_tracing.strip().lower() in {"1", "true", "yes"} and bool(
            self.langsmith_api_key.strip()
        )

    @property
    def s3_enabled(self) -> bool:
        return bool(self.s3_bucket.strip())

    def capabilities(self) -> dict[str, object]:
        """A JSON-serialisable summary of what this process can actually do.

        Surfaced by ``GET /health`` and by the startup log line, so "why is the
        agent behaving like a stub in staging?" has a one-request answer.
        """
        return {
            "environment": self.app_env,
            "llm": {
                "enabled": self.llm_enabled,
                "model": self.llm_model_id,
                "max_calls_per_incident": self.llm_max_calls_per_incident,
            },
            "tracing": {
                "langsmith_enabled": self.langsmith_enabled,
                "project": self.langsmith_project if self.langsmith_enabled else None,
            },
            "storage": {
                "database": _redact_dsn(self.database_url),
                "s3_enabled": self.s3_enabled,
                "log_dir": str(self.pipeline_log_dir),
            },
        }

    def ensure_directories(self) -> None:
        self.pipeline_log_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_estate_file.parent.mkdir(parents=True, exist_ok=True)


def _redact_dsn(dsn: str) -> str:
    """Strip any password out of a DSN before it reaches a log or an HTTP body."""
    if "@" not in dsn or "://" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    credentials, host = rest.rsplit("@", 1)
    user = credentials.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings. Cached — call ``get_settings.cache_clear()`` in tests."""
    settings = Settings()
    settings.ensure_directories()
    return settings
