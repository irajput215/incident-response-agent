"""Configuration: derived capability, loud failures, and secret redaction.

The theme is that a misconfiguration should be *visible*. Every test here checks
a case where the wrong behaviour would be silent — a typo'd log level that
disables logging, a wrong-case backend that quietly means "local", a DSN that
leaks its password into a health endpoint.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import STUB_MODEL, Settings, _redact_dsn


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_empty_configuration_means_the_offline_stub() -> None:
    """The default must be the free, offline, no-key path."""
    settings = make_settings()
    assert settings.llm_model_id == STUB_MODEL
    assert settings.llm_enabled is False
    assert settings.langsmith_enabled is False
    assert settings.s3_enabled is False


def test_a_configured_model_enables_the_real_client() -> None:
    settings = make_settings(llm_model="anthropic/claude-sonnet-4-5")
    assert settings.llm_enabled is True
    assert settings.llm_model_id == "anthropic/claude-sonnet-4-5"


def test_whitespace_only_model_is_treated_as_unset() -> None:
    assert make_settings(llm_model="   ").llm_enabled is False


def test_langsmith_needs_both_the_flag_and_a_key() -> None:
    assert make_settings(langsmith_tracing="true").langsmith_enabled is False
    assert make_settings(langsmith_api_key="lsv2-x").langsmith_enabled is False
    assert (
        make_settings(langsmith_tracing="true", langsmith_api_key="lsv2-x").langsmith_enabled
        is True
    )
    assert make_settings(langsmith_tracing="false", langsmith_api_key="k").langsmith_enabled is False


def test_a_bad_log_level_fails_loudly_at_startup() -> None:
    """Silently ignoring an invalid level is how logging turns itself off."""
    with pytest.raises(ValidationError) as excinfo:
        make_settings(log_level="verbose")
    assert "log_level" in str(excinfo.value)

    # Case-insensitive is fine; the point is that it is validated, not ignored.
    assert make_settings(log_level="debug").log_level == "DEBUG"


def test_an_impossible_pool_size_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(db_pool_max=0)


def test_capabilities_report_what_the_process_actually_has() -> None:
    capabilities = make_settings().capabilities()
    assert capabilities["llm"] == {
        "enabled": False,
        "model": STUB_MODEL,
        "max_calls_per_incident": 25,
    }
    assert capabilities["tracing"]["langsmith_enabled"] is False
    assert capabilities["storage"]["s3_enabled"] is False


def test_capabilities_cannot_drift_from_the_runtime_decision() -> None:
    """`/health` and the client factory must agree; both read `llm_enabled`."""
    settings = make_settings(llm_model="openai/gpt-4o-mini")
    assert settings.capabilities()["llm"]["enabled"] == settings.llm_enabled is True


def test_passwords_never_reach_a_log_or_a_health_response() -> None:
    redacted = _redact_dsn("postgresql://alice:hunter2@db.internal:5432/incidents")
    assert "hunter2" not in redacted
    assert redacted == "postgresql://alice:***@db.internal:5432/incidents"


def test_redaction_leaves_a_passwordless_dsn_alone() -> None:
    assert _redact_dsn("postgresql://localhost:5432/ai_incidents") == (
        "postgresql://localhost:5432/ai_incidents"
    )


def test_capabilities_redact_the_configured_database_url() -> None:
    settings = make_settings(database_url="postgresql://bob:secret@localhost/ai_incidents")
    assert "secret" not in str(settings.capabilities())
