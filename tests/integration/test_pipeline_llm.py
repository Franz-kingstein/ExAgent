"""Full-pipeline integration with the real local SLM (Phase 8).

Requires the Ollama service with qwen3.5:4b; skipped automatically otherwise.
The deterministic stages are covered by tests/e2e/test_pipeline.py; this adds
the end-to-end path including the SLM rationale and the validation guardrail.
"""

import urllib.request

import pytest

from trace_xai import TraceXAIPipeline
from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.rationale.provider import OllamaRationaleProvider


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ollama_up(),
                                reason="Local Ollama not reachable")


class TestPipelineWithRealSLM:
    def test_end_to_end_success_task(self):
        pipeline = TraceXAIPipeline(
            agent=RuleBasedAgent(BankEnvironment.from_dataset()),
            rationale_provider=OllamaRationaleProvider(model="qwen3.5:4b"),
        )
        _, report = pipeline.run("Transfer ₹10,000 to beneficiary BOB-001.")

        # all 15 report sections present
        text = report.to_text()
        for i in range(1, 16):
            assert f"[{i}]" in text
        # the SLM path stays valid and never gets blocked on this trace
        assert report.validation.overall_validation_status.value in (
            "SUPPORTED", "INSUFFICIENT_EVIDENCE")
        assert report.validation.contradicted_claim_count == 0
        assert report.validation.guardrail_decision.value != "BLOCK"
        assert report.rationale.llm_used is True
        assert report.llm_used is True  # pipeline-level flag reflects the stage
        # every stage timed
        assert set(report.timings_ms) == {
            "decision", "deviation", "counterfactual", "rationale", "validation"}
