"""The LLM layer: one interface, two very different implementations.

``build_llm()`` is the only place that decides which one you get, and the rule
is a single line: a configured model means a real client, an empty
``LLM_MODEL`` means the offline heuristic analyst.

Keeping the decision in one function is what makes the offline guarantee
auditable. There is exactly one place to read to answer "could this process
spend money?" — and ``Settings.llm_enabled`` is the same rule, so the answer in
``/health`` cannot drift from the answer at runtime.
"""
from __future__ import annotations

from app.config import Settings, get_settings
from app.llm.base import (
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    extract_json_object,
    render_schema_instruction,
)
from app.llm.heuristic import HeuristicLLM, detect_category, detect_severity
from app.llm.litellm_client import LiteLLMClient
from app.observability import Metrics


def build_llm(
    settings: Settings | None = None,
    *,
    metrics: Metrics | None = None,
) -> LLMClient:
    """Return the client this configuration asks for.

    Real model when ``LLM_MODEL`` is set; otherwise the deterministic analyst.
    The stub is not an error state — it is the default, and it is what CI uses.
    """
    settings = settings or get_settings()
    if settings.llm_enabled:
        return LiteLLMClient(settings, metrics=metrics)
    return HeuristicLLM()


__all__ = [
    "HeuristicLLM",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "LiteLLMClient",
    "Message",
    "ToolCall",
    "ToolSpec",
    "build_llm",
    "detect_category",
    "detect_severity",
    "extract_json_object",
    "render_schema_instruction",
]
