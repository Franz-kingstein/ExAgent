"""Rationale package: SLM-based user-facing rationale layer over Q1-Q3.

The model is optional and replaceable (see provider.py); with no model the
generator composes the same five sections deterministically from evidence.
No component here adds facts: claims must cite the supplied evidence packet.
"""

from .context import EvidenceEntry, RationaleContext, build_context
from .generator import RationaleGenerator, generate_rationale
from .models import ClaimLabel, RationaleClaim, StructuredRationale
from .provider import (
    EchoRationaleProvider,
    OpenAICompatProvider,
    OllamaRationaleProvider,
    ProviderError,
    RationaleProvider,
)
from .template import SYSTEM_PROMPT, render_prompt

__all__ = [
    "build_context",
    "RationaleContext",
    "EvidenceEntry",
    "RationaleGenerator",
    "generate_rationale",
    "ClaimLabel",
    "RationaleClaim",
    "StructuredRationale",
    "RationaleProvider",
    "OllamaRationaleProvider",
    "OpenAICompatProvider",
    "EchoRationaleProvider",
    "ProviderError",
    "SYSTEM_PROMPT",
    "render_prompt",
]
