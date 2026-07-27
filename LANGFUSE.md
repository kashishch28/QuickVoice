# Langfuse Integration

This fork adds [Langfuse](https://github.com/langfuse/langfuse) tracing to QuickVoice's voice-call pipeline, so every AI phone call can be automatically traced and evaluated.

## Why this approach

QuickVoice's LLM calls happen inside LiveKit's `AgentSession` runtime (in `apps/ai`), not via a directly-callable OpenAI/Anthropic client. So instead of wrapping an SDK call, this integration attaches to the same event stream the app already uses internally for transcripts (`conversation_item_added`), plus the LLM plugin's own `metrics_collected` event for token/latency data.

Each LLM turn is correlated to its metrics precisely via `speech_id` — an ID LiveKit stamps on both the emitted `LLMMetrics` and the resulting `ChatMessage`, rather than guessing by event order.

## What gets traced

For every call, a single Langfuse **trace** is created (keyed by `call_id`), containing:

- One **generation** per LLM turn: the user's message, the agent's reply, model name, token usage (input/output/total), latency, and time-to-first-token
- Trace-level metadata: `agent_id`, `organization_id`, `direction`, `provider`, `llm_model`
- Optional **scores**: when the agent records a mid-call evaluation (via the existing `record_call_evaluation` tool), it's also pushed to Langfuse as a boolean/numeric/categorical score on the trace

## Privacy

QuickVoice already supports a per-agent `zero_pii_retention` flag that suppresses transcript/recording storage. This integration respects it: when set, generation `input`/`output` text is replaced with a redaction placeholder, while model, token, and latency metadata are still recorded.

## Setup

1. Install the new dependency (already added to `apps/ai/requirements.txt`):
```bash
   cd apps/ai
   pip install -r requirements.txt
```

2. Add these environment variables to `apps/ai/.env.dev` (all optional — the integration is fully disabled unless `LANGFUSE_ENABLED=true`, with zero behavior change and no new outbound network calls when left off):
```bash
   LANGFUSE_ENABLED=true
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
   LANGFUSE_SAMPLE_RATE=1.0                    # optional, for high-volume sampling
```

3. Get keys: sign up at [cloud.langfuse.com](https://cloud.langfuse.com) (or self-host via `docker compose up` from the [Langfuse repo](https://github.com/langfuse/langfuse)), create a project, and copy its public/secret keys.

## Verifying it works

Run the unit tests (no network or Langfuse instance required):
```bash
cd apps/ai
python -m pytest tests/test_observability_handler.py -v
```
Expected: `19 passed`.

Then, with `LANGFUSE_ENABLED=true` and valid keys set, run the app and make a test call. In the Langfuse dashboard, open the **Traces** tab — you should see a trace named `quickvoice.call`, tagged `quickvoice`/`<provider>`, containing one generation per turn.

## Files changed

| File | Change |
|---|---|
| `apps/ai/handlers/observability_handler.py` | **New.** `CallTraceRecorder` — the core Langfuse integration |
| `apps/ai/tests/test_observability_handler.py` | **New.** 19 unit tests |
| `apps/ai/main.py` | Wires the recorder into the call's start/attach/shutdown lifecycle |
| `apps/ai/requirements.txt` | Adds `langfuse>=4,<5` |
| `apps/ai/.env.dev.example` | Documents the new `LANGFUSE_*` env vars |

## Possible follow-ups

- Export LiveKit's own OpenTelemetry `gen_ai.*` spans directly to Langfuse's OTLP endpoint as a fallback, in case the `metrics_collected`/`speech_id` contract changes in a future LiveKit release
- Add an integration test that exercises `entrypoint()`'s wiring end-to-end against a fake `AgentSession`
