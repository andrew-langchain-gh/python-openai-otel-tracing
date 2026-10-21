"""Offline integration tests: real SDK parsing, real OTel spans, mocked HTTP."""

import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

import httpx2
import requests
from openai import OpenAI, RateLimitError, APITimeoutError
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

import joke_agent as agent
import otel_setup

JOKE = "Why did the database leave? It had too many commitments."
TOOL_CALL = {
    "id": "fc_test", "type": "function_call", "call_id": "call_style",
    "name": "get_joke_style", "arguments": "{}", "status": "completed",
}
REASONING = {
    "id": "rs_test", "type": "reasoning", "summary": [],
    "encrypted_content": "opaque-reasoning-never-export",
}


def response(output, **updates):
    data = {
        "id": "resp_test", "object": "response", "created_at": 1,
        "model": "gpt-test-snapshot", "status": "completed", "output": output,
        "usage": {
            "input_tokens": 21, "output_tokens": 17, "total_tokens": 38,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    }
    data.update(updates)
    return data


def text_output(text=JOKE):
    return [{"id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": text, "annotations": []}]}]


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider(shutdown_on_exit=False)
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.addCleanup(self.provider.shutdown)
        self.tracer_patch = patch.object(otel_setup, "tracer", self.provider.get_tracer("test"))
        self.tracer_patch.start()
        self.addCleanup(self.tracer_patch.stop)
        self.requests = []

    def client(self, responses=None, error=None):
        pending = iter(responses or [response([REASONING, TOOL_CALL]), response(text_output())])

        def handler(request):
            self.requests.append(json.loads(request.content))
            if error is not None:
                raise error(request)
            item = next(pending)
            if isinstance(item, tuple):
                return httpx2.Response(item[0], json=item[1])
            return httpx2.Response(200, json=item)

        client = OpenAI(api_key="test-only", base_url="https://example.invalid:8443/v1",
                        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)), max_retries=0)
        self.addCleanup(client.close)
        return client

    def spans(self):
        return self.exporter.get_finished_spans()

    def test_real_tool_roundtrip_and_span_parentage(self):
        self.assertEqual(agent.tell_joke("  databases  ", self.client()), JOKE)
        self.assertEqual(len(self.requests), 2)
        first, second = self.requests
        self.assertEqual(first["tool_choice"], {"type": "function", "name": "get_joke_style"})
        self.assertEqual(second["tool_choice"], "none")
        self.assertEqual(second["input"][-1], {
            "type": "function_call_output", "call_id": "call_style", "output": "a short pun",
        })
        self.assertEqual(second["input"][1], REASONING)
        for request in self.requests:
            self.assertFalse(request["store"])
            self.assertFalse(request["parallel_tool_calls"])
            self.assertEqual(request["include"], ["reasoning.encrypted_content"])
        spans = self.spans()
        self.assertEqual(len(spans), 7)
        root = next(span for span in spans if span.parent is None)
        self.assertEqual(root.name, "invoke_agent joke-agent")
        for span in spans:
            self.assertEqual(span.context.trace_id, root.context.trace_id)
            if span is not root:
                self.assertEqual(span.parent.span_id, root.context.span_id)
        models = [span for span in spans if span.kind == SpanKind.CLIENT]
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].attributes["gen_ai.response.finish_reasons"], ("tool_calls",))
        self.assertEqual(models[1].attributes["gen_ai.response.finish_reasons"], ("stop",))
        for span in models:
            attrs = span.attributes
            self.assertEqual(span.name, f"generate_content {agent.DEFAULT_MODEL}")
            self.assertEqual(attrs["gen_ai.provider.name"], "openai")
            self.assertEqual(attrs["openai.api.type"], "responses")
            self.assertEqual(attrs["langsmith.span.kind"], "llm")
            self.assertEqual(attrs["server.address"], "example.invalid")
            self.assertEqual(attrs["server.port"], 8443)
            self.assertEqual(attrs["gen_ai.response.model"], "gpt-test-snapshot")
            self.assertEqual(attrs["gen_ai.usage.input_tokens"], 21)
            self.assertEqual(attrs["gen_ai.usage.output_tokens"], 17)
            self.assertEqual(attrs["gen_ai.usage.cache_read.input_tokens"], 3)
            self.assertEqual(attrs["gen_ai.usage.reasoning.output_tokens"], 2)
        tool = next(span for span in spans if span.attributes.get("gen_ai.operation.name") == "execute_tool")
        self.assertEqual(tool.attributes["gen_ai.tool.call.id"], "call_style")
        self.assertEqual(tool.attributes["langsmith.span.kind"], "tool")
        self.assertLessEqual(models[0].end_time, tool.start_time)
        self.assertLessEqual(tool.end_time, models[1].start_time)

    def test_content_is_off_by_default(self):
        agent.tell_joke("private-topic-123", self.client())
        serialized = "".join(span.to_json() for span in self.spans())
        for content in ("private-topic-123", JOKE, "opaque-reasoning-never-export", "a short pun"):
            self.assertNotIn(content, serialized)
        for span in self.spans():
            self.assertNotIn("gen_ai.input.messages", span.attributes)
            self.assertNotIn("gen_ai.output.messages", span.attributes)

    def test_content_opt_in_includes_tool_and_its_result(self):
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
        agent.tell_joke("databases", self.client())
        spans = self.spans()
        models = [span for span in spans if span.kind == SpanKind.CLIENT]
        first_output = json.loads(models[0].attributes["gen_ai.output.messages"])
        self.assertEqual(first_output[0]["parts"][0], {
            "type": "tool_call", "id": "call_style", "name": "get_joke_style", "arguments": {},
        })
        second_input = json.loads(models[1].attributes["gen_ai.input.messages"])
        self.assertEqual(second_input[-1]["parts"][0]["response"], "a short pun")
        self.assertEqual(json.loads(models[1].attributes["gen_ai.system_instructions"]), [
            {"type": "text", "content": agent.INSTRUCTIONS},
        ])
        root = next(span for span in spans if span.parent is None)
        self.assertEqual(json.loads(root.attributes["gen_ai.output.messages"])[0]["parts"][0]["content"], JOKE)
        self.assertNotIn("opaque-reasoning-never-export", "".join(span.to_json() for span in spans))

    def test_invalid_topics_never_call_the_api(self):
        for topic in ("   ", "x" * 1001):
            with self.subTest(length=len(topic)):
                self.exporter.clear()
                with self.assertRaises(ValueError):
                    agent.tell_joke(topic, self.client())
                self.assertEqual(len(self.spans()), 2)
                self.assertTrue(all(span.status.status_code == StatusCode.ERROR for span in self.spans()))
        self.assertEqual(self.requests, [])

    def test_api_error_is_propagated_and_redacted(self):
        private = "private-api-body-123"
        client = self.client([(429, {"error": {"message": private, "type": "rate_limit_error"}})])
        with self.assertRaises(RateLimitError):
            agent.tell_joke("databases", client)
        failed = [span for span in self.spans() if span.status.status_code == StatusCode.ERROR]
        self.assertEqual(len(failed), 2)
        for span in failed:
            self.assertTrue(span.attributes["error.type"].endswith("RateLimitError"))
            self.assertEqual(span.events[0].name, "exception")
        self.assertNotIn(private, "".join(span.to_json() for span in self.spans()))

    def test_timeout_is_an_error_span(self):
        client = self.client(error=lambda request: httpx2.ReadTimeout("private", request=request))
        with self.assertRaises(APITimeoutError):
            agent.tell_joke("databases", client)
        model = next(span for span in self.spans() if span.kind == SpanKind.CLIENT)
        self.assertEqual(model.status.status_code, StatusCode.ERROR)

    def test_incomplete_response_never_reaches_tool(self):
        client = self.client([response(text_output("partial"), status="incomplete",
                                       incomplete_details={"reason": "max_output_tokens"})])
        with self.assertRaises(agent.UnexpectedModelResponse):
            agent.tell_joke("databases", client)
        model = next(span for span in self.spans() if span.kind == SpanKind.CLIENT)
        self.assertEqual(model.attributes["gen_ai.response.finish_reasons"], ("length",))
        self.assertEqual(model.status.status_code, StatusCode.ERROR)
        self.assertEqual(len(self.requests), 1)

    def test_tool_arguments_are_validated_and_error_is_traced(self):
        for arguments in ('{"unexpected": 1}', 'not json'):
            with self.subTest(arguments=arguments):
                self.exporter.clear()
                call = dict(TOOL_CALL, arguments=arguments)
                with self.assertRaises(agent.UnexpectedModelResponse):
                    agent.tell_joke("databases", self.client([response([call])]))
                tool = next(span for span in self.spans() if span.attributes.get("langsmith.span.kind") == "tool")
                self.assertEqual(tool.status.status_code, StatusCode.ERROR)

    def test_missing_tool_call_is_rejected(self):
        with self.assertRaises(agent.UnexpectedModelResponse):
            agent.tell_joke("databases", self.client([response(text_output())]))
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.spans()[-1].status.status_code, StatusCode.ERROR)

    def test_empty_or_refusal_only_joke_is_rejected(self):
        refusal = [{"type": "message", "role": "assistant", "id": "msg_refusal", "status": "completed",
                    "content": [{"type": "refusal", "refusal": "Cannot help"}]}]
        for output in ([], text_output("  "), refusal):
            with self.subTest(output=output):
                with self.assertRaises(agent.UnexpectedModelResponse):
                    agent.tell_joke("databases", self.client([response([TOOL_CALL]), response(output)]))
        self.assertEqual(self.spans()[-1].status.status_code, StatusCode.ERROR)

    def test_usage_can_be_absent(self):
        client = self.client([response([TOOL_CALL], usage=None), response(text_output(), usage=None)])
        self.assertEqual(agent.tell_joke("databases", client), JOKE)
        for span in self.spans():
            self.assertNotIn("gen_ai.usage.input_tokens", span.attributes)

    def test_cli_flushes_batch_on_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                exporter = InMemorySpanExporter()
                provider = TracerProvider(shutdown_on_exit=False)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                client = self.client([(429, {"error": {"message": "private-error-body"}})]) if fail else self.client()
                out, err = io.StringIO(), io.StringIO()
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}), \
                     patch.object(agent, "create_tracer_provider", return_value=provider), \
                     patch.object(agent.trace, "set_tracer_provider"), \
                     patch.object(otel_setup, "tracer", provider.get_tracer("cli-test")), \
                     patch.object(agent, "OpenAI", return_value=client) as factory, \
                     patch.object(provider, "shutdown", wraps=provider.shutdown) as shutdown, \
                     redirect_stdout(out), redirect_stderr(err):
                    code = agent.main(["databases"])
                self.assertEqual(code, 1 if fail else 0)
                factory.assert_called_once_with(timeout=30.0, max_retries=1)
                shutdown.assert_called_once()
                self.assertGreater(len(exporter.get_finished_spans()), 0)
                self.assertNotIn("private-error-body", err.getvalue())
                self.assertEqual(out.getvalue(), "" if fail else JOKE + "\n")

    def test_missing_key_exits_cleanly(self):
        err = io.StringIO()
        with patch.object(agent, "create_tracer_provider", return_value=self.provider), \
             patch.object(agent.trace, "set_tracer_provider"), redirect_stderr(err):
            self.assertEqual(agent.main(["databases"]), 1)
        self.assertIn("OPENAI_API_KEY", err.getvalue())


class ExporterTests(unittest.TestCase):
    def test_otlp_endpoints_headers_payload_and_tls(self):
        cases = [
            ({"OTEL_EXPORTER_OTLP_ENDPOINT": "https://example.invalid/otel",
              "OTEL_EXPORTER_OTLP_HEADERS": "x-api-key=test-only,Langsmith-Project=joke-demo"},
             "https://example.invalid/otel/v1/traces", "joke-demo"),
            ({"OTEL_EXPORTER_OTLP_ENDPOINT": "https://ignored.invalid/otel",
              "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "https://example.invalid/custom/traces",
              "OTEL_EXPORTER_OTLP_HEADERS": "Langsmith-Project=ignored",
              "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "x-api-key=test-only,Langsmith-Project=joke%20demo",
              "OTEL_EXPORTER_OTLP_CERTIFICATE": "", "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE": ""},
             "https://example.invalid/custom/traces", "joke demo"),
        ]
        for environment, endpoint, project in cases:
            with self.subTest(endpoint=endpoint):
                session = Mock(spec=requests.Session)
                reply = requests.Response()
                reply.status_code = 200
                reply._content = b""
                session.request.return_value = reply
                out = io.StringIO()
                with patch.dict(os.environ, environment, clear=True), \
                     patch.object(otel_setup, "OTLPSpanExporter", side_effect=lambda: OTLPSpanExporter(session=session)), \
                     redirect_stderr(out):
                    provider = otel_setup.create_tracer_provider()
                    try:
                        with provider.get_tracer("export-test").start_as_current_span("test"):
                            pass
                        self.assertTrue(provider.force_flush())
                    finally:
                        provider.shutdown()
                self.assertEqual(out.getvalue(), "")
                request = session.request.call_args.kwargs
                self.assertEqual(request["url"], endpoint)
                headers = {key.lower(): value for key, value in request["headers"].items()}
                self.assertEqual(headers["langsmith-project"], project)
                self.assertEqual(headers["content-type"], "application/x-protobuf")
                self.assertIs(session.verify, True)
                payload = ExportTraceServiceRequest.FromString(request["data"])
                self.assertEqual(payload.resource_spans[0].scope_spans[0].spans[0].name, "test")
                session.close.assert_called_once()

    def test_console_fallback_and_service_name(self):
        out = io.StringIO()
        with patch.dict(os.environ, {"OTEL_SERVICE_NAME": "test-joke-service"}, clear=True), redirect_stderr(out):
            provider = otel_setup.create_tracer_provider()
            try:
                with provider.get_tracer("test").start_as_current_span("local-span"):
                    pass
                self.assertTrue(provider.force_flush())
            finally:
                provider.shutdown()
        self.assertIn('"name": "local-span"', out.getvalue())
        self.assertIn('"service.name": "test-joke-service"', out.getvalue())


if __name__ == "__main__":
    unittest.main()
