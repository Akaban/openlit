# pylint: disable=broad-exception-caught
"""
Langfuse compatibility layer for OpenLIT.

This module provides wrapper classes that intercept OTel span attribute calls
and automatically add Langfuse-compatible metadata attributes.
"""

import json
from contextlib import contextmanager

from opentelemetry import trace as trace_api
from opentelemetry.trace import INVALID_SPAN_CONTEXT
from opentelemetry.trace.span import NonRecordingSpan


# Maps to langfuse.observation.input (combined as JSON object)
LANGFUSE_INPUT_MAP = {
    "db.collection.name": "collection_name",
    "db.query.text": "query",
    "db.vector.query.top_k": "top_k",
    "db.filter": "filter",
}

# Maps to langfuse.observation.output (combined as JSON object)
LANGFUSE_OUTPUT_MAP = {
    "db.response.output": "output",
}

# Maps directly to langfuse.observation fields (already complete, set immediately)
LANGFUSE_DIRECT_MAP = {
    "gen_ai.workflow.input": "langfuse.observation.input",
    "gen_ai.workflow.output": "langfuse.observation.output",
}

# Maps to langfuse.observation.metadata.* (contextual info only)
LANGFUSE_METADATA_MAP = {
    "gen_ai.workflow.session_id": "session_id",
    "gen_ai.workflow.run_id": "run_id",
    "gen_ai.workflow.name": "workflow_name",
    "gen_ai.workflow.description": "workflow_description",
    "gen_ai.agent.name": "agent_name",
    "db.response.returned_rows": "returned_rows",
}


class LangfuseCompatSpan:
    """Wraps an OTel span to add Langfuse-compatible metadata attributes."""

    def __init__(self, span):
        self._span = span
        self._input_data = {}  # Collect input attributes
        self._output_data = {}  # Collect output attributes

    def set_attribute(self, key, value):
        """Set attribute and also set Langfuse metadata/observation fields if key is mapped."""
        self._span.set_attribute(key, value)

        if value is None:
            return

        # Collect input attributes (will be combined on span exit)
        input_key = LANGFUSE_INPUT_MAP.get(key)
        if input_key:
            self._input_data[input_key] = value
            return

        # Collect output attributes (will be combined on span exit)
        output_key = LANGFUSE_OUTPUT_MAP.get(key)
        if output_key:
            self._output_data[output_key] = value
            return

        # Direct mapping to langfuse observation fields (already complete values)
        direct_key = LANGFUSE_DIRECT_MAP.get(key)
        if direct_key:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str)
            self._span.set_attribute(direct_key, value)
            return

        # Metadata mapping
        meta_key = LANGFUSE_METADATA_MAP.get(key)
        if meta_key:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str)
            self._span.set_attribute(f"langfuse.observation.metadata.{meta_key}", value)

    def set_attributes(self, attributes):
        """Set multiple attributes."""
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def __getattr__(self, name):
        """Delegate all other methods/attributes to the wrapped span."""
        return getattr(self._span, name)

    def _set_input_output(self):
        """Set combined input/output attributes before span closes."""
        if self._input_data:
            self._span.set_attribute(
                "langfuse.observation.input",
                json.dumps(self._input_data, default=str)
            )
        if self._output_data:
            # Output is typically a single value, unwrap if only one key
            if len(self._output_data) == 1:
                output_value = next(iter(self._output_data.values()))
                if isinstance(output_value, (dict, list)):
                    output_value = json.dumps(output_value, default=str)
            else:
                output_value = json.dumps(self._output_data, default=str)
            self._span.set_attribute("langfuse.observation.output", output_value)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._set_input_output()
        return self._span.__exit__(*args)


class LangfuseCompatTracer:
    """Wraps an OTel tracer to return Langfuse-compatible spans."""

    def __init__(self, tracer, suppress_root_spans=True):
        self._tracer = tracer
        self._suppress_root_spans = suppress_root_spans

    def _is_root(self):
        """Check if the current context has no valid parent span."""
        return not trace_api.get_current_span().get_span_context().is_valid

    @contextmanager
    def start_as_current_span(self, name, **kwargs):
        """Start a span and wrap it with Langfuse compatibility."""
        if self._suppress_root_spans and self._is_root():
            noop_span = NonRecordingSpan(INVALID_SPAN_CONTEXT)
            with trace_api.use_span(noop_span, end_on_exit=False):
                yield LangfuseCompatSpan(noop_span)
            return

        with self._tracer.start_as_current_span(name, **kwargs) as span:
            wrapped = LangfuseCompatSpan(span)
            try:
                yield wrapped
            finally:
                wrapped._set_input_output()

    def start_span(self, name, **kwargs):
        """Start a span and wrap it with Langfuse compatibility."""
        if self._suppress_root_spans and self._is_root():
            return LangfuseCompatSpan(NonRecordingSpan(INVALID_SPAN_CONTEXT))

        span = self._tracer.start_span(name, **kwargs)
        return LangfuseCompatSpan(span)

    def __getattr__(self, name):
        """Delegate all other methods/attributes to the wrapped tracer."""
        return getattr(self._tracer, name)


def wrap_tracer_for_langfuse(tracer, suppress_root_spans=True):
    """Wrap a tracer with Langfuse-compatible metadata augmentation."""
    if tracer is None:
        raise ValueError("tracer cannot be None")

    return LangfuseCompatTracer(tracer, suppress_root_spans=suppress_root_spans)
