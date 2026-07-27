import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from handlers import observability_handler as obs


class FakeSpan:
    def __init__(self, **kwargs):
        self.id = "span-1"
        self.trace_id = kwargs.get("trace_context", {}).get("trace_id")
        self.kwargs = kwargs
        self.ended = False
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended = True


class FakeClient:
    def __init__(self):
        self.observations = []
        self.scores = []
        self.flushed = False

    def create_trace_id(self, *, seed=None):
        return f"trace-{seed}"

    def start_observation(self, **kwargs):
        span = FakeSpan(**kwargs)
        self.observations.append(span)
        return span

    def create_score(self, **kwargs):
        self.scores.append(kwargs)

    def flush(self):
        self.flushed = True


class FakePropagationCtx:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *args):
        self.exited = True
        return False


class FakeSession:
    def __init__(self):
        self.handlers = {}

    def on(self, event, callback):
        self.handlers[event] = callback


class FakeLLM:
    def __init__(self):
        self.handlers = {}

    def on(self, event, callback):
        self.handlers[event] = callback


def _install_fake_langfuse_module(propagation_calls):
    fake_module = types.ModuleType("langfuse")

    def propagate_attributes(**kwargs):
        propagation_calls.append(kwargs)
        return FakePropagationCtx(**kwargs)

    fake_module.propagate_attributes = propagate_attributes
    sys.modules["langfuse"] = fake_module


class CallTraceRecorderTests(unittest.TestCase):
    def setUp(self):
        obs.reset_langfuse_client_for_tests()
        self.propagation_calls: list = []
        _install_fake_langfuse_module(self.propagation_calls)
        self.fake_client = FakeClient()
        self._client_patch = patch.object(obs, "get_langfuse_client", return_value=self.fake_client)
        self._client_patch.start()

    def tearDown(self):
        self._client_patch.stop()
        sys.modules.pop("langfuse", None)
        obs.reset_langfuse_client_for_tests()

    def _make_recorder(self, **config_overrides):
        call_context = {
            "call_id": "call-123",
            "agent_id": "agent-1",
            "direction": "inbound",
            "provider": "TWILIO",
        }
        config = {
            "organization_id": "org-1",
            "llm_model": "anthropic/claude-haiku-4-5",
            "first_message": "Hello, how can I help?",
            "zero_pii_retention": False,
        }
        config.update(config_overrides)
        return obs.CallTraceRecorder(call_context=call_context, config=config), call_context, config

    def test_disabled_client_is_a_no_op(self):
        self._client_patch.stop()
        with patch.object(obs, "get_langfuse_client", return_value=None):
            recorder, _, _ = self._make_recorder()
            session = FakeSession()
            llm = FakeLLM()
            recorder.attach(session, llm)
            self.assertEqual(session.handlers, {})
            self.assertEqual(llm.handlers, {})
            recorder.close(reason="test")  # must not raise
        self._client_patch.start()

    def test_start_trace_opens_root_span_with_propagated_attributes(self):
        recorder, call_context, config = self._make_recorder()

        self.assertEqual(len(self.fake_client.observations), 1)
        root_span = self.fake_client.observations[0]
        self.assertEqual(root_span.kwargs["as_type"], "span")
        self.assertEqual(root_span.kwargs["input"]["call_id"], "call-123")
        self.assertEqual(root_span.kwargs["input"]["first_message"], "Hello, how can I help?")

        self.assertEqual(len(self.propagation_calls), 1)
        propagated = self.propagation_calls[0]
        self.assertEqual(propagated["session_id"], "call-123")
        self.assertEqual(propagated["user_id"], "org-1")
        self.assertIn("quickvoice", propagated["tags"])

    def test_zero_pii_retention_redacts_call_input(self):
        recorder, _, _ = self._make_recorder(zero_pii_retention=True)
        root_span = self.fake_client.observations[0]
        self.assertEqual(root_span.kwargs["input"], {"call_id": "call-123"})

    def test_records_generation_with_correlated_llm_metrics(self):
        recorder, _, _ = self._make_recorder()
        session = FakeSession()
        llm = FakeLLM()
        recorder.attach(session, llm)

        llm.handlers["metrics_collected"](
            SimpleNamespace(
                speech_id="speech-1",
                prompt_tokens=42,
                completion_tokens=8,
                total_tokens=50,
                prompt_cached_tokens=0,
                duration=0.9,
                ttft=0.2,
                cancelled=False,
                metadata=SimpleNamespace(model_name="claude-haiku-4-5", model_provider="anthropic"),
            )
        )

        session.handlers["conversation_item_added"](
            SimpleNamespace(item=SimpleNamespace(role="user", text_content="What's my balance?", extra={}))
        )
        session.handlers["conversation_item_added"](
            SimpleNamespace(
                item=SimpleNamespace(
                    role="assistant",
                    text_content="Your balance is $42.",
                    extra={"speech_id": "speech-1"},
                )
            )
        )

        # First observation is the root span; second is the generation.
        self.assertEqual(len(self.fake_client.observations), 2)
        generation = self.fake_client.observations[1]
        self.assertEqual(generation.kwargs["as_type"], "generation")
        self.assertEqual(generation.kwargs["input"], "What's my balance?")
        self.assertEqual(generation.kwargs["output"], "Your balance is $42.")
        self.assertEqual(generation.kwargs["model"], "claude-haiku-4-5")
        self.assertEqual(generation.kwargs["usage_details"]["input"], 42)
        self.assertEqual(generation.kwargs["usage_details"]["output"], 8)
        self.assertTrue(generation.ended)

    def test_generation_recorded_even_without_matching_metrics(self):
        recorder, _, _ = self._make_recorder()
        session = FakeSession()
        llm = FakeLLM()
        recorder.attach(session, llm)

        session.handlers["conversation_item_added"](
            SimpleNamespace(item=SimpleNamespace(role="user", text_content="Hi", extra={}))
        )
        session.handlers["conversation_item_added"](
            SimpleNamespace(
                item=SimpleNamespace(role="assistant", text_content="Hello!", extra={"speech_id": "unknown"})
            )
        )

        generation = self.fake_client.observations[1]
        self.assertIsNone(generation.kwargs["usage_details"])
        self.assertTrue(generation.kwargs["metadata"]["metrics_unavailable"])

    def test_zero_pii_retention_redacts_generation_text_but_keeps_usage(self):
        recorder, _, _ = self._make_recorder(zero_pii_retention=True)
        session = FakeSession()
        llm = FakeLLM()
        recorder.attach(session, llm)

        llm.handlers["metrics_collected"](
            SimpleNamespace(
                speech_id="speech-1",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                prompt_cached_tokens=0,
                duration=0.5,
                ttft=0.1,
                cancelled=False,
                metadata=None,
            )
        )
        session.handlers["conversation_item_added"](
            SimpleNamespace(item=SimpleNamespace(role="user", text_content="secret stuff", extra={}))
        )
        session.handlers["conversation_item_added"](
            SimpleNamespace(
                item=SimpleNamespace(role="assistant", text_content="secret reply", extra={"speech_id": "speech-1"})
            )
        )

        generation = self.fake_client.observations[1]
        self.assertEqual(generation.kwargs["input"], "[REDACTED_ZERO_PII_RETENTION]")
        self.assertEqual(generation.kwargs["output"], "[REDACTED_ZERO_PII_RETENTION]")
        self.assertEqual(generation.kwargs["usage_details"]["input"], 10)

    def test_close_ends_root_span_and_flushes_client(self):
        recorder, _, _ = self._make_recorder()
        recorder.close(reason="participant_disconnected")

        root_span = self.fake_client.observations[0]
        self.assertTrue(root_span.ended)
        self.assertEqual(root_span.updates[-1]["output"]["shutdown_reason"], "participant_disconnected")
        self.assertTrue(self.fake_client.flushed)

    def test_close_is_safe_when_trace_never_started(self):
        self._client_patch.stop()
        with patch.object(obs, "get_langfuse_client", return_value=None):
            recorder, _, _ = self._make_recorder()
            recorder.close(reason="test")  # must not raise
        self._client_patch.start()

    def test_record_evaluation_boolean_becomes_langfuse_boolean_score(self):
        recorder, _, _ = self._make_recorder()
        recorder.record_evaluation("call_resolved", True, description="Caller's issue was resolved")

        self.assertEqual(len(self.fake_client.scores), 1)
        score = self.fake_client.scores[0]
        self.assertEqual(score["name"], "call_resolved")
        self.assertEqual(score["value"], 1.0)
        self.assertEqual(score["data_type"], "BOOLEAN")
        self.assertEqual(score["trace_id"], recorder._trace_id)
        self.assertEqual(score["comment"], "Caller's issue was resolved")

    def test_record_evaluation_numeric_becomes_langfuse_numeric_score(self):
        recorder, _, _ = self._make_recorder()
        recorder.record_evaluation("satisfaction", 4)

        score = self.fake_client.scores[0]
        self.assertEqual(score["value"], 4.0)
        self.assertEqual(score["data_type"], "NUMERIC")

    def test_record_evaluation_string_becomes_categorical_score(self):
        recorder, _, _ = self._make_recorder()
        recorder.record_evaluation("sentiment", "frustrated")

        score = self.fake_client.scores[0]
        self.assertEqual(score["value"], "frustrated")
        self.assertEqual(score["data_type"], "CATEGORICAL")

    def test_record_evaluation_skips_none_values(self):
        recorder, _, _ = self._make_recorder()
        recorder.record_evaluation("call_resolved", None)
        self.assertEqual(self.fake_client.scores, [])

    def test_record_evaluation_skips_blank_identifier(self):
        recorder, _, _ = self._make_recorder()
        recorder.record_evaluation("   ", True)
        self.assertEqual(self.fake_client.scores, [])

    def test_record_evaluation_is_a_no_op_when_client_disabled(self):
        self._client_patch.stop()
        with patch.object(obs, "get_langfuse_client", return_value=None):
            recorder, _, _ = self._make_recorder()
            recorder.record_evaluation("call_resolved", True)  # must not raise
        self._client_patch.start()

    def test_record_evaluation_swallows_client_errors(self):
        recorder, _, _ = self._make_recorder()
        recorder._client.create_score = lambda **_: (_ for _ in ()).throw(RuntimeError("boom"))
        recorder.record_evaluation("call_resolved", True)  # must not raise

    def test_langfuse_client_error_during_start_disables_recorder_without_raising(self):
        broken_client = FakeClient()
        broken_client.start_observation = lambda **_: (_ for _ in ()).throw(RuntimeError("boom"))
        with patch.object(obs, "get_langfuse_client", return_value=broken_client):
            recorder, _, _ = self._make_recorder()
            session = FakeSession()
            llm = FakeLLM()
            recorder.attach(session, llm)  # must not raise
            recorder.close(reason="test")  # must not raise


class GetLangfuseClientTests(unittest.TestCase):
    def setUp(self):
        obs.reset_langfuse_client_for_tests()
        self._env_backup = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)
        obs.reset_langfuse_client_for_tests()

    def test_returns_none_when_disabled(self):
        os.environ["LANGFUSE_ENABLED"] = "false"
        self.assertIsNone(obs.get_langfuse_client())

    def test_returns_none_when_keys_missing(self):
        os.environ["LANGFUSE_ENABLED"] = "true"
        os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
        os.environ.pop("LANGFUSE_SECRET_KEY", None)
        self.assertIsNone(obs.get_langfuse_client())

    def test_is_memoized_across_calls(self):
        os.environ["LANGFUSE_ENABLED"] = "false"
        first = obs.get_langfuse_client()
        second = obs.get_langfuse_client()
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
