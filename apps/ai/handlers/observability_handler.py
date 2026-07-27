"""Langfuse tracing for voice calls.

QuickVoice's LLM calls happen entirely inside LiveKit's ``AgentSession``
runtime (see ``main.py``), not via a directly-callable OpenAI/Anthropic
client. So instead of wrapping an SDK call, this module attaches to the same
``AgentSession`` event stream that ``TranscriptCollector`` already uses
(``conversation_item_added``), plus the LLM plugin's own ``metrics_collected``
event, to build one Langfuse trace per call with one "generation" observation
per completed LLM turn.

Design notes (also covered in the PR description / interview prep):

* Per-turn correlation uses ``speech_id``, which LiveKit stamps on both the
  emitted ``LLMMetrics`` and the resulting assistant ``ChatMessage.extra``.
  This is precise (not "nearest event in time") and survives features like
  preemptive generation that can otherwise reorder events.
* Every Langfuse call is wrapped in try/except and logged at ``warning``
  level only. Observability must never be able to break or delay a live
  call — if Langfuse is unreachable or misconfigured, the call proceeds
  exactly as if this module were absent.
* ``zero_pii_retention`` (an existing per-agent config flag already enforced
  by ``CallFinalizer`` for call logs and recordings) is enforced here too:
  when set, generation ``input``/``output`` are replaced with a redaction
  placeholder while model, token, and latency metadata are still recorded.
* Everything no-ops (returns None / does nothing) unless ``LANGFUSE_ENABLED``
  is explicitly set, so self-hosted forks that don't run Langfuse see zero
  behavior change and no new outbound network calls.
"""

from __future__ import annotations

import os
from typing import Any

from utils.logger import logger

_client: Any = None
_client_initialized = False


def _langfuse_enabled() -> bool:
    return os.getenv("LANGFUSE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _sample_rate() -> float:
    try:
        return float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0"))
    except ValueError:
        return 1.0


def get_langfuse_client() -> Any:
    """Return a lazily-constructed, process-wide Langfuse client, or None.

    Returns None whenever Langfuse is disabled, unconfigured, or the SDK
    fails to import/initialize. Callers must treat None as "tracing is off"
    and never raise on its behalf.
    """
    global _client, _client_initialized
    if _client_initialized:
        return _client
    _client_initialized = True

    if not _langfuse_enabled():
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        logger.warning(
            "[langfuse] LANGFUSE_ENABLED is true but LANGFUSE_PUBLIC_KEY/"
            "LANGFUSE_SECRET_KEY are missing; tracing stays disabled"
        )
        return None

    try:
        from langfuse import Langfuse
    except Exception as error:
        logger.warning("[langfuse] SDK import failed, tracing disabled: {}", str(error))
        return None

    try:
        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            sample_rate=_sample_rate(),
        )
    except Exception as error:
        logger.warning("[langfuse] client initialization failed, tracing disabled: {}", str(error))
        _client = None
    return _client


def reset_langfuse_client_for_tests() -> None:
    """Test-only hook to force re-initialization of the module-level client."""
    global _client, _client_initialized
    _client = None
    _client_initialized = False


class CallTraceRecorder:
    """Traces a single voice call as a Langfuse trace with per-turn generations."""

    def __init__(self, *, call_context: dict[str, Any], config: dict[str, Any]):
        self._call_context = call_context
        self._config = config
        self._zero_pii = bool(config.get("zero_pii_retention"))
        self._pending_llm_metrics: dict[str, Any] = {}
        self._last_user_text: str | None = None
        self._turn_count = 0
        self._root_span: Any = None
        self._propagation_ctx: Any = None
        self._trace_id: str | None = None

        self._client = get_langfuse_client()
        if self._client is None:
            return

        try:
            self._start_trace()
        except Exception as error:
            logger.warning("[langfuse] failed to start call trace: {}", str(error))
            self._client = None
            self._root_span = None

    def _start_trace(self) -> None:
        from langfuse import propagate_attributes

        call_id = str(self._call_context.get("call_id") or "unknown-call")
        self._trace_id = self._client.create_trace_id(seed=call_id)

        metadata = {
            "agent_id": self._call_context.get("agent_id"),
            "organization_id": self._config.get("organization_id"),
            "direction": self._call_context.get("direction"),
            "provider": self._call_context.get("provider"),
            "llm_model": self._config.get("llm_model"),
            "zero_pii_retention": self._zero_pii,
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}

        self._propagation_ctx = propagate_attributes(
            user_id=str(self._config.get("organization_id") or "unknown-org"),
            session_id=call_id,
            tags=["quickvoice", str(self._call_context.get("provider") or "unknown-provider")],
            metadata=metadata,
            trace_name="quickvoice.call",
        )
        self._propagation_ctx.__enter__()

        self._root_span = self._client.start_observation(
            trace_context={"trace_id": self._trace_id},
            name="quickvoice.call",
            as_type="span",
            input=self._safe_call_input(),
        )

    def _safe_call_input(self) -> dict[str, Any]:
        if self._zero_pii:
            return {"call_id": self._call_context.get("call_id")}
        return {
            "call_id": self._call_context.get("call_id"),
            "first_message": self._config.get("first_message"),
        }

    def attach(self, session: Any, llm: Any) -> "CallTraceRecorder":
        """Subscribe to the LLM plugin's metrics and the session's transcript events.

        Must be called with the *same* llm object that was passed into
        ``AgentSession(...)`` — attaching to a different instance (e.g. one
        rebuilt later) will never fire.
        """
        if self._client is None:
            return self
        try:
            llm.on("metrics_collected", self._on_llm_metrics)
            session.on("conversation_item_added", self._on_conversation_item)
        except Exception as error:
            logger.warning("[langfuse] failed to attach listeners: {}", str(error))
        return self

    def _on_llm_metrics(self, metrics: Any) -> None:
        speech_id = getattr(metrics, "speech_id", None)
        if not speech_id:
            return
        self._pending_llm_metrics[speech_id] = metrics

    def _on_conversation_item(self, event: Any) -> None:
        if self._client is None:
            return

        item = getattr(event, "item", None)
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None)
        if callable(text):
            text = text()
        text = str(text or "").strip()
        if not text:
            return

        if role == "user":
            self._last_user_text = text
            return

        if role != "assistant":
            return

        extra = getattr(item, "extra", None) or {}
        speech_id = extra.get("speech_id") if isinstance(extra, dict) else None
        metrics = self._pending_llm_metrics.pop(speech_id, None) if speech_id else None
        self._record_generation(agent_text=text, metrics=metrics)

    def _record_generation(self, *, agent_text: str, metrics: Any) -> None:
        self._turn_count += 1
        user_text = self._last_user_text
        self._last_user_text = None

        model_name = self._config.get("llm_model")
        usage_details = None
        turn_metadata: dict[str, Any] = {}
        if metrics is not None:
            usage_details = {
                "input": getattr(metrics, "prompt_tokens", 0),
                "output": getattr(metrics, "completion_tokens", 0),
                "total": getattr(metrics, "total_tokens", 0),
                "cache_read_input_tokens": getattr(metrics, "prompt_cached_tokens", 0),
            }
            plugin_metadata = getattr(metrics, "metadata", None)
            model_name = (getattr(plugin_metadata, "model_name", None) if plugin_metadata else None) or model_name
            turn_metadata = {
                "duration_seconds": getattr(metrics, "duration", None),
                "ttft_seconds": getattr(metrics, "ttft", None),
                "cancelled": getattr(metrics, "cancelled", None),
            }
        else:
            # No matching LLMMetrics (e.g. the plugin doesn't emit them, or the
            # speech was cancelled before metrics fired). We still record the
            # turn's text so the trace stays complete, just without token/
            # latency numbers.
            turn_metadata = {"metrics_unavailable": True}

        if self._zero_pii:
            input_payload: Any = "[REDACTED_ZERO_PII_RETENTION]"
            output_payload: Any = "[REDACTED_ZERO_PII_RETENTION]"
        else:
            input_payload = user_text
            output_payload = agent_text

        try:
            generation = self._client.start_observation(
                trace_context=self._trace_context(),
                name=f"llm-turn-{self._turn_count}",
                as_type="generation",
                input=input_payload,
                output=output_payload,
                model=model_name,
                usage_details=usage_details,
                metadata=turn_metadata,
            )
            generation.end()
        except Exception as error:
            logger.warning("[langfuse] failed to record generation: {}", str(error))

    def _trace_context(self) -> dict[str, Any]:
        if self._root_span is not None:
            return {"trace_id": self._trace_id, "parent_span_id": self._root_span.id}
        return {"trace_id": self._trace_id}

    def record_evaluation(self, identifier: str, value: Any, *, description: str | None = None) -> None:
        """Push a QuickVoice mid-call evaluation result as a Langfuse score.

        Mirrors the values `CallMetadataCollector.record_evaluation` already
        normalizes (bool / categorical string / numeric), so this can be
        called with the exact same arguments right after that method, from
        the `record_call_evaluation` function tool. `None` values (the
        normalized form of "n/a"/"unknown") are skipped: a evaluation the
        model couldn't determine isn't a meaningful score.
        """
        if self._client is None or self._trace_id is None:
            return
        if value is None:
            return

        name = str(identifier or "").strip()
        if not name:
            return

        if isinstance(value, bool):
            data_type = "BOOLEAN"
            score_value: float | str = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            data_type = "NUMERIC"
            score_value = float(value)
        else:
            data_type = "CATEGORICAL"
            score_value = str(value)

        try:
            self._client.create_score(
                name=name,
                value=score_value,
                trace_id=self._trace_id,
                session_id=str(self._call_context.get("call_id") or ""),
                data_type=data_type,
                comment=description or None,
            )
        except Exception as error:
            logger.warning("[langfuse] failed to record evaluation score '{}': {}", name, str(error))

    def close(self, *, reason: str) -> None:
        """End the call trace. Safe to call even if the trace never started."""
        if self._client is None:
            return
        try:
            if self._root_span is not None:
                self._root_span.update(output={"shutdown_reason": reason, "turn_count": self._turn_count})
                self._root_span.end()
        except Exception as error:
            logger.warning("[langfuse] failed to close call trace: {}", str(error))
        try:
            if self._propagation_ctx is not None:
                self._propagation_ctx.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self._client.flush()
        except Exception as error:
            logger.warning("[langfuse] flush failed: {}", str(error))
