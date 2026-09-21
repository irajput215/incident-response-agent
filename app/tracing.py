"""LangSmith tracing: configured from ``Settings``, and inert without a key.

Lives at the top level rather than under ``app.evaluation`` on purpose: tracing is
an observability concern that *every* layer uses, and putting it under evaluation
created a genuine import cycle (nodes → evaluation → harness → agents → nodes).

The design rule is that **the same code path runs on every machine**. Tracing is
configured here and is a no-op when no key is present, so the traced and
untraced paths cannot drift — which is the usual failure mode of an optional
observability integration: it works on the author's laptop and silently does
nothing (or crashes) in CI.

What gets traced, once a key is present:

* every graph node, via ``@traceable``, so a run reads as
  ``triage → investigate → assess_root_cause → …`` with timings;
* every LLM call, via LiteLLM's native LangSmith callback, so token counts and
  prompts appear as child spans automatically;
* tool calls already carry their own timing and are attached to the node span's
  metadata through the repository records.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

from app.config import Settings, get_settings
from app.observability import get_logger, log

_log = get_logger("app.tracing")

F = TypeVar("F", bound=Callable[..., Any])

# Env vars LangSmith reads. Kept in one place so "is tracing on?" has a single
# definition that matches what the SDK actually looks at.
_TRUTHY = {"1", "true", "yes", "on"}


def configure_tracing(settings: Settings | None = None) -> bool:
    """Push the configured LangSmith credentials into the environment.

    Returns whether tracing is enabled. Called once at startup, before any
    LangGraph or LiteLLM call, because both read the environment at import or
    first-use time.
    """
    settings = settings or get_settings()

    if settings.langsmith_enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        # LiteLLM has native LangSmith support; enabling the callback is what
        # makes model calls appear as child spans with token usage, rather than
        # as opaque gaps between nodes.
        try:
            import litellm

            callbacks = list(getattr(litellm, "success_callback", []) or [])
            if "langsmith" not in callbacks:
                callbacks.append("langsmith")
            litellm.success_callback = callbacks
        except Exception as exc:  # pragma: no cover - litellm always importable
            log(_log, 30, "litellm_tracing_callback_failed", error=str(exc)[:200])

        log(
            _log,
            20,
            "tracing_enabled",
            project=settings.langsmith_project,
            endpoint=settings.langsmith_endpoint,
        )
        return True

    # Explicitly clear, so a stale value in the shell cannot make one process
    # trace into another team's project.
    for key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        os.environ.pop(key, None)
    log(_log, 10, "tracing_disabled")
    return False


def tracing_enabled() -> bool:
    """Whether the SDK will actually emit anything right now."""
    return (
        os.environ.get("LANGSMITH_TRACING", "").lower() in _TRUTHY
        or os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in _TRUTHY
    ) and bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))


def traceable(*decorator_args: Any, **decorator_kwargs: Any) -> Callable[[F], F]:
    """``langsmith.traceable``, but safe to import and call unconditionally.

    Node methods are decorated with this at class-definition time. Importing
    ``langsmith.traceable`` directly would be fine too, but wrapping it here
    means the project has exactly one place that knows about the SDK, and a
    future change of tracing backend is a change to one function.
    """
    try:
        from langsmith import traceable as _traceable
    except Exception:  # pragma: no cover - langsmith is a hard dependency
        def passthrough(fn: F) -> F:
            return fn

        if decorator_args and callable(decorator_args[0]) and not decorator_kwargs:
            return decorator_args[0]  # used bare, as @traceable
        return passthrough

    return _traceable(*decorator_args, **decorator_kwargs)


__all__ = ["configure_tracing", "traceable", "tracing_enabled"]
