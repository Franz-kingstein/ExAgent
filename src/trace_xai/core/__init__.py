"""TRACE-XAI core: the evidence spine.

This package holds the trace/evidence primitives that every later phase
(decision explanation, deviation detection, counterfactuals, validation)
consumes. Nothing here fabricates information: missing data stays explicitly
missing (None/null). See docs/safety.md.
"""

from .trace import (
    ErrorInfo,
    EventType,
    ExecutionTrace,
    Provenance,
    ProvenanceSource,
    TraceEvent,
)
from .recorder import ExecutionRecorder

__all__ = [
    "ErrorInfo",
    "EventType",
    "ExecutionTrace",
    "Provenance",
    "ProvenanceSource",
    "TraceEvent",
    "ExecutionRecorder",
]
