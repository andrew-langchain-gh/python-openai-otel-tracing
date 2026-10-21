# Python joke agent with OpenTelemetry

A small CLI that tells a joke about your topic. Ordinary Python orchestrates the
OpenAI Responses API calls and a real function tool; decorators trace each step.
It uses no agent framework or LangSmith SDK. Traces reach LangSmith using standard
OTLP over HTTP/protobuf.

## Setup and run

Requires Python 3.11 or newer and an OpenAI API key supplied through your environment.

```sh
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
# Supply OPENAI_API_KEY through your shell or secret manager.
python joke_agent.py "programming"
```

Without a configured OTLP endpoint, spans appear on stderr; stdout contains only
the joke. Run `python joke_agent.py` to enter a topic interactively. Override the
default `gpt-5.4-mini` with `--model MODEL` or `OPENAI_MODEL`. The handoff's proposed
`gpt-4.1-mini` is now deprecated, so this example uses a supported small GPT model.

## Send traces to LangSmith

Supply `LANGSMITH_API_KEY` through your environment, then run:

```sh
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.smith.langchain.com/otel"
export OTEL_EXPORTER_OTLP_HEADERS="x-api-key=${LANGSMITH_API_KEY},Langsmith-Project=python-joke-agent"
export OTEL_SERVICE_NAME="python-joke-agent"
python joke_agent.py "OpenTelemetry"
```

Open the `python-joke-agent` project in LangSmith. For EU, APAC, AWS, or self-hosted
LangSmith, replace the base endpoint with the endpoint for your deployment.
Header values containing spaces, commas, or other reserved characters must be
percent-encoded. A project name with hyphens needs no encoding.

`OTEL_EXPORTER_OTLP_ENDPOINT` is a **base** URL: the HTTP exporter appends
`/v1/traces`. Alternatively, `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` takes the **complete**
URL, such as `https://api.smith.langchain.com/otel/v1/traces`, and overrides the base.
The trace-specific `OTEL_EXPORTER_OTLP_TRACES_HEADERS` overrides general headers.
No `LANGSMITH_TRACING` or `LANGSMITH_OTEL_ENABLED` flags are needed.

Add `--console-traces` to inspect spans on stderr alongside remote export. The CLI
flushes and shuts down the batch processors in `finally`, including after errors.
An exporter error is reported by OTel on stderr; a successful joke does not itself
prove that a remote backend accepted the trace.

## Trace structure and content

```text
invoke_agent joke-agent                 INTERNAL / chain
  validate_topic                        INTERNAL / chain
  build_prompt                          INTERNAL / prompt
  generate_content gpt-5.4-mini          CLIENT   / llm (requests the tool)
  execute_tool get_joke_style            INTERNAL / tool
  generate_content gpt-5.4-mini          CLIENT   / llm (uses the tool result)
  extract_joke                          INTERNAL / parser
```

The first model request forces `get_joke_style`, an intentionally trivial tool
that returns `a short pun`. Python validates and executes the tool, then passes
its result to a second model request. Tool calling is disabled in the second
request, so each run is bounded to two model calls and one tool execution.
Encrypted reasoning items are carried between requests for stateless reasoning
models, but are never put in traces.

`otel_setup.traced` handles context, span completion, and error status. The
application only adds explicit attributes to the current span; it never opens or
closes spans. Errors propagate to parent spans with `error.type` and a sanitized
exception event. API error bodies, exception messages, stack traces, function
arguments, and client objects are not automatically captured.

By default, traces contain model identifiers, server address, response ID, timing,
finish reasons, and input/output token usage (plus cached/reasoning tokens when
reported). To see the topic, prompt, instructions, and joke, explicitly enable:

```sh
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
python joke_agent.py "databases"
```

Content is then sent to the configured trace destination, including the local
console exporter. It uses the current structured `gen_ai.input.messages`,
`gen_ai.output.messages`, and `gen_ai.system_instructions` attributes, JSON encoded
for Python OTel. Keep capture disabled for topics you do not want in telemetry.
The topic is always sent to OpenAI to generate the joke. `store=False` disables
Responses API response storage; it does not configure organization data policies.

The model span uses `gen_ai.provider.name=openai`,
`gen_ai.operation.name=generate_content`, and `openai.api.type=responses`.
The legacy `gen_ai.system=openai` alias and `langsmith.span.kind` hints support
LangSmith's documented provider/run-type mapping. GenAI conventions are still in
Development; the selected behavior follows the documents linked below as reviewed
on October 21, 2026. Completed tool-call Responses map to `tool_calls`, final text to `stop`; incomplete token-limited
responses map to `length`. Incomplete responses and empty/refusal-only output are
errors, and partial text is never printed as a successful joke. One model span
covers the SDK request, including its optional retry.

## Tests

```sh
python -m unittest discover -s tests -v
```

The tests use the real OpenAI SDK with its HTTP mock transport and OTel's in-memory
exporter. They make no external API calls and need no credentials. They exercise
parentage, semantic attributes, content opt-in, usage, errors, endpoint/header
configuration, and flushing on exit. The runtime dependencies are bounded in
`requirements.txt`; tests need no additional packages.

## References

- [OpenAI text generation](https://developers.openai.com/api/docs/guides/text)
- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
- [LangSmith OpenTelemetry integration and attribute mapping](https://docs.langchain.com/langsmith/trace-with-opentelemetry)
- [OTel GenAI client conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md)
- [OTel OpenAI conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/openai.md)

OpenAI and OpenTelemetry are official, actively maintained Apache-2.0 projects.
The exporter minimum includes the TLS-verification fix in OTel 1.45.0.
