"""GUI + real local Ollama: the SLM toggle end-to-end.

Skipped automatically when the local daemon is unreachable. The unit tests
cover the toggle wiring with a fake provider; this test proves the real
provider reports genuine token usage into the GUI's metrics panel.
"""

import json
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import app as gui  # noqa: E402


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ollama_up(),
                                reason="Local Ollama not reachable")


def test_gui_slm_toggle_records_real_provider_usage():
    out = gui.run_scenario(list(gui.SCENARIOS)[0], "", gui._fresh_state(),
                           "normal", True)
    report = out[2]["report"]
    assert report.llm_used is True, \
        "with Ollama up the rationale must come from the local SLM"
    assert report.model_usage, "provider-reported usage must be recorded"
    usage = report.model_usage[0]
    assert usage.provider.startswith("ollama")
    assert usage.model == "qwen3.5:4b"
    assert usage.prompt_tokens and usage.completion_tokens
    assert usage.duration_ms and usage.duration_ms > 0

    # the metrics panel shows the real counts and stays honest about cost
    metrics = gui.metrics_view(out[2], False, 0, 0, False)
    assert f"**{usage.prompt_tokens}**" in metrics
    assert f"**{usage.completion_tokens}**" in metrics
    assert "no per-token price" in metrics

    # with evaluator-supplied pricing the estimate is labelled ESTIMATED
    priced = gui.metrics_view(out[2], True, 0.0001, 0.0002, False)
    assert "Pricing supplied" in priced
    assert "estimated" in priced.lower()
