"""Small, explicit OpenTelemetry setup and a synchronous function decorator."""

from __future__ import annotations

import os
import sys
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode

P = ParamSpec("P")
R = TypeVar("R")
tracer = trace.get_tracer("python-joke-agent", "1.0.0")


def capture_content() -> bool:
    return os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "").lower() == "true"


def traced(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, str] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Let OTel manage span lifetimes; record errors without API bodies or arguments."""
    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            with tracer.start_as_current_span(
                name, kind=kind, attributes=attributes,
                record_exception=False, set_status_on_exception=False,
            ) as span:
                try:
                    return function(*args, **kwargs)
                except Exception as exc:
                    error_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
                    span.set_attribute("error.type", error_type)
                    span.set_status(Status(StatusCode.ERROR, error_type))
                    # Exception messages/stack traces can include content or secrets.
                    span.add_event("exception", {
                        "exception.type": error_type,
                        "exception.escaped": True,
                    })
                    raise
        return wrapped
    return decorate


def create_tracer_provider(*, console: bool = False) -> TracerProvider:
    """Use standard OTLP environment settings, falling back to stderr locally."""
    provider = TracerProvider(
        resource=Resource.create({
            "service.name": os.getenv("OTEL_SERVICE_NAME") or "python-joke-agent",
        }),
        shutdown_on_exit=False,
    )
    remote = bool(
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    try:
        if remote:
            # The exporter handles base vs trace-specific endpoints and headers.
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        if console or not remote:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter(out=sys.stderr)))
    except Exception:
        provider.shutdown()
        raise
    return provider
