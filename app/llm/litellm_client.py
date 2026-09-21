"""LiteLLM-backed :class:`LLMClient` — one code path for every provider.

LiteLLM is the reason this project can claim provider-agnosticism honestly:
Anthropic, OpenAI, DeepSeek and a local Ollama model all arrive through the same
``completion()`` call, so switching model is a config change and the evaluation
suite can compare two providers without touching agent code.

Three things here are deliberate and worth defending:

**1. ``litellm`` is imported lazily.** It is a large import. The offline stub
path — which is what CI and the unit tests use — never pays for it.

**2. ``timeout`` is always explicit.** ``httpx`` has no default timeout, so an
un-specified network call waits forever. A hung 2am agent is worse than a failed
one, so the timeout comes from settings and is never omitted.

**3. ``structured()`` repairs rather than gives up.** Small models routinely emit
JSON that is 95% right — a trailing comment, a stray enum value. One bounded
repair attempt, with the validation error quoted back, converts a large fraction
of those into successes. The attempt count is bounded because an unbounded
repair loop is just a slower way to hang.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any

from pydantic import ValidationError

from app.config import Settings
from app.llm.base import (
    LLMError,
    LLMResponse,
    Message,
    TModel,
    ToolCall,
    ToolSpec,
    extract_json_object,
    render_schema_instruction,
)
from app.observability import METRICS, Metrics, get_logger, log

_log = get_logger("app.llm.litellm")

# A single repair attempt. Enough to fix a formatting slip, bounded so a
# model that cannot produce valid JSON fails fast instead of burning budget.
MAX_REPAIR_ATTEMPTS = 1


class LiteLLMClient:
    """Calls a real model through LiteLLM."""

    def __init__(
        self,
        settings: Settings,
        *,
        metrics: Metrics | None = None,
        model: str | None = None,
    ) -> None:
        self._settings = settings
        self._metrics = metrics if metrics is not None else METRICS
        self._model = model or settings.llm_model_id

    @property
    def model(self) -> str:
        return self._model

    # --- provider wiring ---------------------------------------------------
    def _provider(self) -> str:
        return self._model.split("/", 1)[0] if "/" in self._model else ""

    def _auth_kwargs(self) -> dict[str, Any]:
        """Map the configured credentials onto LiteLLM's ``api_key``/``api_base``.

        Passed explicitly rather than relying on LiteLLM reading the environment,
        because configuration here flows through ``Settings`` (and therefore
        ``.env``), and having two sources of truth for a credential is how you
        end up debugging a deployment that reads neither.
        """
        provider = self._provider()
        if provider == "anthropic" and self._settings.anthropic_api_key:
            return {"api_key": self._settings.anthropic_api_key}
        if provider == "openai" and self._settings.openai_api_key:
            return {"api_key": self._settings.openai_api_key}
        if provider == "deepseek" and self._settings.deepseek_api_key:
            return {"api_key": self._settings.deepseek_api_key}
        if provider == "ollama":
            # Local server: no credential, but the base URL must be explicit.
            return {"api_base": self._settings.ollama_api_base}
        return {}

    # --- the one call ------------------------------------------------------
    def _completion(self, messages: list[Message], **overrides: Any) -> Any:
        import litellm  # lazy: the stub path never imports this

        # Let each provider drop parameters it does not support, so one request
        # shape works across all of them.
        litellm.drop_params = True

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._settings.llm_temperature
            if overrides.get("temperature") is None
            else overrides["temperature"],
            "timeout": self._settings.llm_timeout_s,  # never omitted: httpx has no default
            "num_retries": self._settings.llm_max_retries,
            **self._auth_kwargs(),
        }
        if overrides.get("tools"):
            kwargs["tools"] = [t.to_openai() if isinstance(t, ToolSpec) else t for t in overrides["tools"]]
            kwargs["tool_choice"] = "auto"
        if overrides.get("json_mode"):
            kwargs["response_format"] = {"type": "json_object"}

        started = perf_counter()
        try:
            return litellm.completion(**kwargs)
        finally:
            self._metrics.observe("llm.latency_ms", (perf_counter() - started) * 1000)

    # --- LLMClient ---------------------------------------------------------
    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        try:
            raw = self._completion(
                messages, tools=tools, temperature=temperature, json_mode=json_mode
            )
        except Exception as exc:  # provider errors are varied; normalise them all
            self._metrics.incr("llm.error")
            log(_log, 40, "llm_call_failed", model=self._model, error=str(exc)[:300])
            raise LLMError(f"{type(exc).__name__} calling {self._model}: {exc}") from exc

        response = self._parse_response(raw)
        self._record_usage(response)
        return response

    def structured(
        self,
        messages: list[Message],
        schema: type[TModel],
        *,
        temperature: float | None = None,
    ) -> TModel:
        """Ask for ``schema``, validate it, and repair once if malformed."""
        prompt = [*messages, {"role": "user", "content": render_schema_instruction(schema)}]
        attempt_messages = list(prompt)
        last_error: str = ""

        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            try:
                raw = self._completion(
                    attempt_messages, temperature=temperature, json_mode=True
                )
            except Exception as exc:
                self._metrics.incr("llm.error")
                raise LLMError(f"{type(exc).__name__} calling {self._model}: {exc}") from exc

            response = self._parse_response(raw)
            self._record_usage(response)

            try:
                payload = extract_json_object(response.text)
                return schema.model_validate(payload)
            except (LLMError, ValidationError) as exc:
                last_error = str(exc)
                self._metrics.incr("llm.structured_invalid")
                if attempt >= MAX_REPAIR_ATTEMPTS:
                    break
                log(
                    _log,
                    30,
                    "llm_structured_repair",
                    model=self._model,
                    schema=schema.__name__,
                    error=last_error[:300],
                )
                # Quote the failure back and demand only the corrected object.
                # Cheaper and more reliable than re-asking the original question.
                attempt_messages = [
                    *messages,
                    {"role": "user", "content": render_schema_instruction(schema)},
                    {"role": "assistant", "content": response.text[:2000]},
                    {
                        "role": "user",
                        "content": (
                            "That response did not validate. Error:\n"
                            f"{last_error[:800]}\n\n"
                            "Return ONLY the corrected JSON object."
                        ),
                    },
                ]

        self._metrics.incr("llm.structured_failed")
        raise LLMError(
            f"{self._model} failed to produce a valid {schema.__name__} "
            f"after {MAX_REPAIR_ATTEMPTS + 1} attempt(s): {last_error[:300]}"
        )

    # --- response handling -------------------------------------------------
    def _parse_response(self, raw: Any) -> LLMResponse:
        try:
            choice = raw.choices[0]
            message = choice.message
        except (AttributeError, IndexError, KeyError) as exc:
            raise LLMError(f"unexpected response shape from {self._model}: {raw!r:.300}") from exc

        text = _as_text(getattr(message, "content", None))
        tool_calls = [
            _parse_tool_call(call) for call in (getattr(message, "tool_calls", None) or [])
        ]
        usage = getattr(raw, "usage", None)

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            model=str(getattr(raw, "model", self._model) or self._model),
            tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def _record_usage(self, response: LLMResponse) -> None:
        self._metrics.incr("llm.calls")
        self._metrics.incr("llm.tokens_in", response.tokens_in)
        self._metrics.incr("llm.tokens_out", response.tokens_out)


def _as_text(content: Any) -> str:
    """Normalise a message body.

    Some providers return a string, others a list of content blocks. Handling
    both here means no node ever has to know which provider produced the text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(getattr(block, "text", "")))
        return "".join(parts)
    return str(content)


def _parse_tool_call(call: Any) -> ToolCall:
    """Convert a provider tool call into our :class:`ToolCall`.

    Unparseable arguments are captured in ``parse_error`` rather than raised, so
    the tool layer can record a clean validation failure instead of the whole
    investigation dying on a ``JSONDecodeError``.
    """
    import json

    call_id = str(getattr(call, "id", "") or "")
    function = getattr(call, "function", None)
    if function is None and isinstance(call, dict):
        function = call.get("function", {})
    name = str(getattr(function, "name", "") or (function or {}).get("name", "") or "")
    raw_args = getattr(function, "arguments", None)
    if raw_args is None and isinstance(function, dict):
        raw_args = function.get("arguments")

    if isinstance(raw_args, dict):
        return ToolCall(id=call_id, name=name, arguments=raw_args)
    if not raw_args:
        return ToolCall(id=call_id, name=name, arguments={})
    try:
        parsed = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError) as exc:
        return ToolCall(
            id=call_id, name=name, arguments={}, parse_error=f"invalid JSON arguments: {exc}"
        )
    if not isinstance(parsed, dict):
        return ToolCall(
            id=call_id,
            name=name,
            arguments={},
            parse_error=f"tool arguments must be an object, got {type(parsed).__name__}",
        )
    return ToolCall(id=call_id, name=name, arguments=parsed)
