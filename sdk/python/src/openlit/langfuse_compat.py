# pylint: disable=broad-exception-caught
"""
Langfuse compatibility layer for OpenLIT.

This module provides wrapper classes that intercept OTel span attribute calls
and automatically add Langfuse-compatible metadata attributes.
"""

import json
from contextlib import contextmanager


# Mapping from OTel semconv to Langfuse metadata keys
LANGFUSE_METADATA_MAP = {
    "gen_ai.workflow.session_state": "session_state",
    "gen_ai.workflow.session_id": "session_id",
    "gen_ai.workflow.run_id": "run_id",
    "gen_ai.workflow.name": "workflow_name",
    "gen_ai.workflow.description": "workflow_description",
    "db.collection.name": "collection_name",
    "db.query.text": "query",
    "db.query.summary": "query_summary",
    "db.vector.query.top_k": "top_k",
    "db.filter": "query_filter",
    "gen_ai.agent.name": "agent_name",
    # Add more as needed
}


class LangfuseCompatSpan:
    """Wraps an OTel span to add Langfuse-compatible metadata attributes."""

    def __init__(self, span):
        self._span = span

    def set_attribute(self, key, value):
        """Set attribute and also set Langfuse metadata if key is mapped."""
        self._span.set_attribute(key, value)

        # Check if this key should be mapped to Langfuse metadata
        lf_key = LANGFUSE_METADATA_MAP.get(key)
        if lf_key and value is not None:
            # Serialize complex values
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str)
            self._span.set_attribute(f"langfuse.observation.metadata.{lf_key}", value)

    def set_attributes(self, attributes):
        """Set multiple attributes."""
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def __getattr__(self, name):
        """Delegate all other methods/attributes to the wrapped span."""
        return getattr(self._span, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self._span.__exit__(*args)


class LangfuseCompatTracer:
    """Wraps an OTel tracer to return Langfuse-compatible spans."""

    def __init__(self, tracer):
        self._tracer = tracer

    @contextmanager
    def start_as_current_span(self, name, **kwargs):
        """Start a span and wrap it with Langfuse compatibility."""
        with self._tracer.start_as_current_span(name, **kwargs) as span:
            yield LangfuseCompatSpan(span)

    def start_span(self, name, **kwargs):
        """Start a span and wrap it with Langfuse compatibility."""
        span = self._tracer.start_span(name, **kwargs)
        return LangfuseCompatSpan(span)

    def __getattr__(self, name):
        """Delegate all other methods/attributes to the wrapped tracer."""
        return getattr(self._tracer, name)


def wrap_tracer_for_langfuse(tracer):
    """Wrap a tracer with Langfuse-compatible metadata augmentation."""
    if tracer is None:
        raise ValueError("tracer cannot be None")

    return LangfuseCompatTracer(tracer)
