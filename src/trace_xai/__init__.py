"""TRACE-XAI: evidence-grounded explainability for AI-agent behavior.

Research sandbox. All banking data is synthetic. See docs/safety.md.
"""

from .pipeline import FinalXAIReport, TraceXAIPipeline, analyze_execution

__version__ = "0.1.0"

SYNTHETIC_DATA_NOTICE = (
    "All financial data in this project is SYNTHETIC/FAKE. "
    "No real banking systems, accounts, or credentials are used."
)

__all__ = [
    "FinalXAIReport",
    "TraceXAIPipeline",
    "analyze_execution",
    "SYNTHETIC_DATA_NOTICE",
]
