# pylint: disable=broad-exception-caught
"""Tests for the Langfuse compatibility layer's root span suppression."""

import threading

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace.span import NonRecordingSpan

# Import the module file directly to avoid pulling in openlit.__init__ and all its heavy deps
import importlib.util as _ilu
import pathlib as _pathlib

_src = _pathlib.Path(__file__).resolve().parent.parent / "src" / "openlit" / "langfuse_compat.py"
_spec = _ilu.spec_from_file_location("langfuse_compat", _src)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LangfuseCompatTracer = _mod.LangfuseCompatTracer
wrap_tracer_for_langfuse = _mod.wrap_tracer_for_langfuse


class InMemorySpanExporter(SpanExporter):
    """Minimal in-memory exporter for tests (avoids version-dependent import paths)."""

    def __init__(self):
        self._spans = []
        self._lock = threading.Lock()

    def export(self, spans):
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self):
        with self._lock:
            return list(self._spans)

    def shutdown(self):
        pass


@pytest.fixture(autouse=True)
def _reset_tracer_provider():
    """Reset the global TracerProvider before each test."""
    provider = TracerProvider()
    trace_api.set_tracer_provider(provider)
    yield


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter):
    provider = trace_api.get_tracer_provider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test")


class TestRootSpansSuppressedByDefault:
    """Root spans should be suppressed by default (suppress_root_spans=True)."""

    def test_start_as_current_span_suppressed(self, tracer, exporter):
        compat = LangfuseCompatTracer(tracer)
        with compat.start_as_current_span("chat gemini-2.5-flash-lite") as span:
            span.set_attribute("key", "value")

        assert len(exporter.get_finished_spans()) == 0

    def test_start_span_suppressed(self, tracer, exporter):
        compat = LangfuseCompatTracer(tracer)
        span = compat.start_span("chat gemini-2.5-flash-lite")
        span.set_attribute("key", "value")
        span._span.end()

        assert len(exporter.get_finished_spans()) == 0

    def test_suppressed_span_is_non_recording(self, tracer, exporter):
        compat = LangfuseCompatTracer(tracer)
        span = compat.start_span("chat gemini-2.5-flash-lite")
        assert isinstance(span._span, NonRecordingSpan)

    def test_body_still_executes(self, tracer, exporter):
        compat = LangfuseCompatTracer(tracer)
        side_effect = []

        with compat.start_as_current_span("anything") as span:
            side_effect.append("executed")
            span.set_attribute("key", "value")

        assert side_effect == ["executed"]
        assert len(exporter.get_finished_spans()) == 0


class TestChildSpansAlwaysAllowed:
    """Child spans should always pass through, even with suppress_root_spans=True."""

    def test_child_span_exported(self, tracer, exporter):
        compat = LangfuseCompatTracer(tracer)

        # Simulate a parent span (e.g., Langfuse observation) already in context
        with tracer.start_as_current_span("langfuse-observation"):
            with compat.start_as_current_span("chat gemini-2.5-flash-lite") as child:
                child.set_attribute("key", "value")

        spans = exporter.get_finished_spans()
        names = {s.name for s in spans}
        assert "langfuse-observation" in names
        assert "chat gemini-2.5-flash-lite" in names
        assert len(spans) == 2

    def test_child_start_span_exported(self, tracer, exporter):
        compat = LangfuseCompatTracer(tracer)

        with tracer.start_as_current_span("langfuse-observation"):
            child = compat.start_span("some-llm-call")
            child.set_attribute("key", "value")
            child._span.end()

        spans = exporter.get_finished_spans()
        names = {s.name for s in spans}
        assert "langfuse-observation" in names
        assert "some-llm-call" in names


class TestSuppressRootSpansDisabled:
    """When suppress_root_spans=False, all spans pass through (backward compat)."""

    def test_root_span_passes(self, tracer, exporter):
        compat = LangfuseCompatTracer(tracer, suppress_root_spans=False)
        with compat.start_as_current_span("anything-goes") as span:
            span.set_attribute("key", "value")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "anything-goes"

    def test_start_span_passes(self, tracer, exporter):
        compat = LangfuseCompatTracer(tracer, suppress_root_spans=False)
        span = compat.start_span("chat gemini-2.5-flash-lite")
        span.set_attribute("key", "value")
        span._span.end()

        assert len(exporter.get_finished_spans()) == 1


class TestWrapTracerForLangfuse:
    """Test the wrap_tracer_for_langfuse factory function."""

    def test_default_suppresses_root(self, tracer, exporter):
        compat = wrap_tracer_for_langfuse(tracer)
        with compat.start_as_current_span("stray-root"):
            pass
        assert len(exporter.get_finished_spans()) == 0

    def test_allows_children(self, tracer, exporter):
        compat = wrap_tracer_for_langfuse(tracer)
        with tracer.start_as_current_span("parent"):
            with compat.start_as_current_span("child"):
                pass
        assert len(exporter.get_finished_spans()) == 2

    def test_suppress_false(self, tracer, exporter):
        compat = wrap_tracer_for_langfuse(tracer, suppress_root_spans=False)
        with compat.start_as_current_span("root"):
            pass
        assert len(exporter.get_finished_spans()) == 1

    def test_raises_on_none_tracer(self):
        with pytest.raises(ValueError, match="tracer cannot be None"):
            wrap_tracer_for_langfuse(None)
