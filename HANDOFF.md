# Python joke agent with OpenTelemetry — implementation complete

Updated October 21, 2026.

## User request and constraints

- Pure Python orchestration, OpenAI Python SDK, no agent framework or LangSmith SDK.
- Tell a joke about a user topic; trace application steps using function decorators.
- Follow GenAI conventions and export standard OTLP/HTTP to LangSmith.
- User additionally requested a simple actual tool call so its trace is visible.
- Never add agent co-authors, modify changelogs, or hand-edit generated files.
- Prefer quality, simplicity, robustness, scalability, and maintainability.

## Implemented files

- `joke_agent.py`: CLI, topic validation, prompt building, two Responses API calls,
  actual `get_joke_style` tool execution, and final joke extraction.
- `otel_setup.py`: synchronous span decorator, sanitized errors, OTLP batch export,
  stderr console fallback, and provider creation.
- `requirements.txt`: bounded official OpenAI and OTel dependencies.
- `tests/test_joke_agent.py`: 15 offline integration tests, no extra dependencies.
- `.gitignore` and `README.md`: setup, LangSmith routing, trace tree, content policy,
  test instructions, and official reference links.

The first request forces the no-argument `get_joke_style` tool, which returns
`a short pun`. Python executes it and sends its output to the second request,
which disables tool calling and generates the joke. The workflow is bounded.
The default is `gpt-5.4-mini`; `--model` or `OPENAI_MODEL` overrides it. The earlier
proposed GPT-4.1 mini is deprecated, so it was not used as the default.

Application spans are `invoke_agent joke-agent`, `validate_topic`, `build_prompt`,
two `generate_content {model}` CLIENT spans, `execute_tool get_joke_style`, and
`extract_joke`. LangSmith displays the tool using its tool name.

Content capture is off unless
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`. Model/tool message
attributes use JSON-encoded current structured GenAI schemas. API error bodies,
exception messages, stack traces, arguments/client objects, and encrypted reasoning
are not implicitly captured. Encrypted reasoning is preserved only between model
requests because `store=False` is used. Errors have sanitized events and error.type.

## Environment and dependency installation

The session declares a read-only filesystem sandbox and restricted networking;
this describes agent tool permissions, not the project's actual filesystem mode.
Approved escalated commands successfully created project files and `.venv` and
accessed the network. No global Codex configuration was changed.

- Python: local `.venv`, CPython 3.13.16.
- Installed/tested: OpenAI 3.36.0, OTel API/SDK/HTTP exporter 1.45.1.
- The official Socket Firewall standalone download URL returned 404. Its supported
  npm installer worked at `/tmp/langsmith-safety-helper/node_modules/.bin/sfw`.
  The launcher needed its executable bit restored. Verified Socket Firewall 1.15.4.
- Application dependencies were installed through Socket Firewall.
- Registry and source metadata were checked: official, maintained Apache-2.0 projects.
- Verified GHSA-6469-8p9h-mmvm and used the fixed exporter release.
- OSV querybatch audit checked all 28 installed distributions and returned no advisories.
- The directory remains outside Git; no git initialization, commit, or push was requested.

## Validation results

```sh
.venv/bin/python -B -m unittest discover -s tests -v
```

All 15 tests passed. They use the real OpenAI SDK with httpx2.MockTransport and real
OTel spans, in-memory export, and mocked OTLP HTTP requests. Coverage includes
real function-call round-trip, ordering/parentage, tool IDs, semantic attributes,
usage, content on/off, reasoning exclusion, topic/tool argument validation, API
errors/timeouts, incomplete and empty/refusal responses, optional usage, CLI
flushing on success/error, endpoint/header precedence, and TLS verification with
empty certificate environment settings.

A real OpenAI smoke request was attempted using OPENAI_API_KEY already present
after sourcing ~/.zshrc. OpenAI returned AuthenticationError; no key or response
body was displayed. LangSmith accepted the resulting four-span error trace with
HTTP 200 in project `python-joke-agent`.

To let the user inspect a complete tool trace immediately, a second demonstration
used the same real SDK/mock HTTP fixtures as the tests and exported to a separate
LangSmith project. It is explicitly labeled `DEMO: mocked OpenAI tool round-trip`
with mock-openai/offline-demo tags and demo.model_transport=mock. Read-back confirmed
all eight spans (seven application spans plus the demo wrapper), two `llm` runs,
one `get_joke_style` tool run, correct parent IDs, and zero errors.

Demo project:
https://smith.langchain.com/o/ff74f6b5-f214-4385-bf4f-60fd0d94da58/projects/p/5c722e7e-f120-4bc9-9e8d-976e08f99972

LangSmith root run/trace ID: `00000000-0000-0000-e919-94f37185dc18`.
Native OTel trace ID: `9174e22f77ce2105a8bf43b68f67a354`.
LangSmith derives its run/trace UUIDs from span IDs; querying by the native OTel
trace UUID did not match. `/runs/query` accepts a scalar `trace` UUID, not an array.
Querying by the project session ID found and verified the demonstration.

## Running

Supply a working OPENAI_API_KEY through the environment (never paste or print it).
Then follow README.md:

```sh
source .venv/bin/activate
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.smith.langchain.com/otel"
export OTEL_EXPORTER_OTLP_HEADERS="x-api-key=${LANGSMITH_API_KEY},Langsmith-Project=python-joke-agent"
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
python joke_agent.py "OpenTelemetry"
```

Omit OTLP endpoint settings for console-only traces; add --console-traces for both.
The only outstanding external issue is the existing OpenAI credential rejection.
The successful complete demonstration used mocked model responses, not live GPT.
