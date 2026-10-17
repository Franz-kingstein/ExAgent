"""Phase 6 tests: SLM rationale layer (user-facing, evidence-grounded).

Required cases:
1. Evidence complete          4. Requirement violated
2. Evidence missing           5. Counterfactual available
3. Evidence conflicts         6. Counterfactual unavailable

Plus: the SLM must not invent evidence (quarantine tests), must not be
required (deterministic fallback), and the prompt must never ask for CoT.
"""

import json

import pytest

from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.core.recorder import ExecutionRecorder
from trace_xai.rationale import (
    ClaimLabel,
    EchoRationaleProvider,
    RationaleGenerator,
    build_context,
    generate_rationale,
    render_prompt,
)
from trace_xai.rationale.models import StructuredRationale
from trace_xai.rationale.template import SYSTEM_PROMPT


class FakeSLM:
    """Scripted provider: returns preset responses in order, records calls."""

    name = "fake-slm"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)


def _valid_output(**overrides) -> str:
    """A well-formed, fully-grounded model response for the 75k trace."""
    payload = {
        "decision_rationale": "The transfer of INR 75000 exceeded the limit.",
        "deviation_explanation": "The execution violated one registered requirement.",
        "counterfactual_summary": "Reducing the amount to the limit would succeed.",
        "evidence_summary": "The tool input and pre-action limits were recorded.",
        "uncertainty_statement": "Assessments carry confidence below 1.0.",
        "claims": [
            {"text": "Amount 75000 exceeded the per-transaction limit.",
             "label": "EVIDENCE", "evidence_refs": ["REQ-AMT-001"]},
            {"text": "Pre-action balance was 250000.",
             "label": "FACT", "evidence_refs": ["E1"]},
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.fixture
def agent() -> RuleBasedAgent:
    return RuleBasedAgent(BankEnvironment.from_dataset())


# ------------------------------------------------------- 1. complete evidence
class TestCase1EvidenceComplete:
    def test_context_contains_all_ten_sections(self, agent):
        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        ctx = build_context(trace)
        rendered = ctx.render()
        for marker in ("[1. USER REQUIREMENT]", "[2. AGENT ACTION]",
                       "[3. REQUIREMENT MATCHING (Q1)]", "[4. DEVIATIONS (Q2)]",
                       "[5. TRAJECTORY ANOMALIES (Q2)]", "[6. COUNTERFACTUALS (Q3)]",
                       "[7. EVIDENCE INDEX]", "[8. UNCERTAINTY]",
                       "[9. MISSING INFORMATION]", "[10. PROVENANCE]",
                       "[ALLOWED CITATION REFS]"):
            assert marker in rendered
        assert ctx.evidence, "evidence index must be populated"
        assert "REQ-AMT-001" in ctx.allowed_refs()

    def test_full_pipeline_with_valid_model_output(self, agent):
        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        slm = FakeSLM([_valid_output()])
        rationale = RationaleGenerator(provider=slm).generate(trace)

        assert isinstance(rationale, StructuredRationale)
        assert rationale.llm_used is True
        assert rationale.grounded is True
        assert rationale.unsupported_claims == []
        assert rationale.provider == "fake-slm"
        assert rationale.claims[0].evidence_refs == ["REQ-AMT-001"]
        assert rationale.claims[1].evidence_refs == ["E1"]
        # prompt was structured, deterministic, and packet-bearing
        system, user = slm.calls[0]
        assert "[ALLOWED CITATION REFS]" in user
        assert system == SYSTEM_PROMPT

    def test_prompt_never_requests_chain_of_thought(self, agent):
        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        system, user = render_prompt(build_context(trace))
        combined = (system + user).lower()
        for forbidden in ("reveal your reasoning", "show your thinking",
                          "explain your reasoning step", "internal reasoning process"):
            assert forbidden not in combined
        assert "hidden reasoning" in system  # the rule *forbidding* it is present


# --------------------------------------------------------- 2. missing evidence
def _stateless_trace():
    recorder = ExecutionRecorder(
        task_id="task-llm-style", task_text="Transfer ₹10,000 to beneficiary BOB-001.",
        agent_id="qwen3.5:4b",
    )
    recorder.record_task_received()
    recorder.record_tool_invocation(
        action="transfer", tool_name="transfer",
        tool_input={"amount_inr": 10_000, "beneficiary_id": "BOB-001"},
        state_before=None,
    )
    recorder.record_tool_result(
        action="transfer", tool_name="transfer",
        tool_output={"success": False, "reason_code": "INSUFFICIENT_FUNDS"},
        success=False,
    )
    recorder.record_outcome("INSUFFICIENT_FUNDS", success=False)
    return recorder.finish()


class TestCase2EvidenceMissing:
    def test_context_marks_missing_state_explicitly(self):
        ctx = build_context(_stateless_trace())
        assert any("pre-action state" in m for m in ctx.missing)
        assert "UNKNOWN (not recorded)" in ctx.render()

    def test_fallback_rationale_states_insufficient_evidence(self):
        rationale = generate_rationale(_stateless_trace())
        assert rationale.llm_used is False
        assert rationale.grounded is True
        assert any(ClaimLabel.UNKNOWN is c.label for c in rationale.claims)
        assert "UNKNOWN" in rationale.uncertainty_statement

    def test_model_output_with_invented_evidence_is_quarantined(self):
        # the model hallucinates a state value that was never recorded
        invented = _valid_output(claims=[
            {"text": "Balance before the transfer was 999999.",
             "label": "FACT", "evidence_refs": ["E7"]},  # E7 does not exist
        ])
        rationale = RationaleGenerator(provider=FakeSLM([invented])).generate(
            _stateless_trace()
        )
        assert rationale.grounded is False
        assert rationale.claims == []          # nothing ungrounded survives
        assert rationale.unsupported_claims    # the attempt is visible
        assert "999999" in rationale.unsupported_claims[0] or \
               "unresolvable refs" in rationale.unsupported_claims[0]

    def test_invented_id_inside_prose_section_is_detected(self):
        # hallucination hides in the *section text*, not the claims array
        prose = _valid_output(
            evidence_summary="Balance was 1,234,000 per evidence E99 and REQ-FAKE-001."
        )
        rationale = RationaleGenerator(provider=FakeSLM([prose])).generate(
            _stateless_trace()
        )
        assert rationale.grounded is False
        assert any("E99" in issue for issue in rationale.unsupported_claims)
        assert any("REQ-FAKE-001" in issue for issue in rationale.unsupported_claims)


# -------------------------------------------------------- 3. conflicting evidence
class TestCase3EvidenceConflicts:
    def _conflicting_trace(self):
        env = BankEnvironment.from_dataset()
        recorder = ExecutionRecorder(
            task_id="task-conflict", task_text="Transfer ₹5,000 to beneficiary BOB-001.",
            agent_id="rule-based-agent",
        )
        recorder.record_task_received()
        inv = recorder.record_tool_invocation(
            action="transfer", tool_name="transfer",
            tool_input={"amount_inr": 5_000, "beneficiary_id": "BOB-001"},
            state_before=env.summary(),
        )
        first = recorder.record_tool_result(
            action="transfer", tool_name="transfer",
            tool_output={"success": True, "reason_code": "OK", "account_id": "ACC-SYN-001"},
            success=True,
        )
        recorder.record_tool_result(
            action="transfer", tool_name="transfer",
            tool_output={"success": False, "reason_code": "INSUFFICIENT_FUNDS",
                         "account_id": "ACC-SYN-001"},
            success=False,
            conflict_with_event_id=first.event_id,
        )
        recorder.record_outcome("CONFLICT", success=None)
        return recorder.finish()

    def test_context_surfaces_conflict_without_resolving_it(self):
        from trace_xai.explain import explain_decision

        trace = self._conflicting_trace()
        decision = explain_decision(trace)
        conflicting = [a for a in decision.requirements if a.conflicting_evidence]
        assert conflicting, "conflicting evidence must reach the decision layer"

        ctx = build_context(trace, decision=decision)
        rendered = ctx.render()
        assert "conflict" in rendered.lower() or "conflicting" in rendered.lower()

    def test_fallback_rationale_does_not_silently_resolve_conflict(self):
        rationale = generate_rationale(self._conflicting_trace())
        combined = (rationale.deviation_explanation + rationale.uncertainty_statement
                    + " ".join(c.text for c in rationale.claims)).lower()
        assert "conflict" in combined


# ----------------------------------------------------- 4. requirement violated
class TestCase4RequirementViolated:
    def test_decision_rationale_names_violated_requirement(self, agent):
        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        rationale = generate_rationale(trace)
        combined = (rationale.decision_rationale + " "
                    + " ".join(c.text for c in rationale.claims))
        assert "REQ-AMT-001" in combined
        assert "VIOLATED" in combined

    def test_claims_are_citable_in_context(self, agent):
        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        ctx = build_context(trace)
        rationale = RationaleGenerator().generate(trace, context=ctx)
        allowed = ctx.allowed_refs()
        for claim in rationale.claims:
            for ref in claim.evidence_refs:
                assert ref in allowed


# ------------------------------------------------- 5. counterfactual available
class TestCase5CounterfactualAvailable:
    def test_counterfactual_summary_mentions_alternative_and_replay(self, agent):
        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        rationale = generate_rationale(trace)
        ctx = build_context(trace)
        assert ctx.counterfactuals, "counterfactuals must be present in the packet"
        combined = (rationale.counterfactual_summary + " "
                    + " ".join(c.text for c in rationale.claims))
        assert "CF-0001" in combined
        assert "replay" in combined.lower()


# ----------------------------------------------- 6. counterfactual unavailable
class TestCase6CounterfactualUnavailable:
    def test_clean_success_has_no_counterfactuals_and_says_so(self, agent):
        _, trace = agent.run_traced("Transfer ₹10,000 to beneficiary BOB-001.")
        ctx = build_context(trace)
        assert ctx.counterfactuals == [] or all(
            line.startswith("note:") for line in ctx.counterfactuals
        )
        rationale = generate_rationale(trace)
        combined = (rationale.counterfactual_summary + " "
                    + " ".join(c.text for c in rationale.claims)).lower()
        assert "no counterfactual" in combined or "nothing to" in combined


# ------------------------------------------------------ provider replacement
class TestProviderReplaceability:
    def test_generator_works_without_any_provider(self, agent):
        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        rationale = generate_rationale(trace)  # provider=None
        assert rationale.llm_used is False
        assert rationale.provider == "deterministic"
        assert rationale.grounded is True
        assert "REQ-AMT-001" in rationale.decision_rationale

    def test_invalid_json_falls_back_deterministically(self, agent):
        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        rationale = RationaleGenerator(provider=FakeSLM(["not json at all"])).generate(trace)
        assert rationale.llm_used is False
        assert any("not valid" in n for n in rationale.notes)
        assert "REQ-AMT-001" in rationale.decision_rationale

    def test_provider_error_falls_back_deterministically(self, agent):
        from trace_xai.rationale import ProviderError

        class BrokenProvider:
            name = "broken"

            def generate(self, system_prompt, user_prompt):
                raise ProviderError("daemon down")

        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        rationale = RationaleGenerator(provider=BrokenProvider()).generate(trace)
        assert rationale.llm_used is False
        assert any("provider failed" in n for n in rationale.notes)

    def test_empty_sections_are_backfilled_from_evidence(self, agent):
        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        empty = json.dumps({
            "decision_rationale": "", "deviation_explanation": "   ",
            "counterfactual_summary": None, "evidence_summary": 5,
            "uncertainty_statement": [], "claims": [],
        })
        rationale = RationaleGenerator(provider=FakeSLM([empty])).generate(trace)
        # unusable sections never survive as blank: evidence composition fills in
        assert rationale.decision_rationale.strip()
        assert rationale.deviation_explanation.strip()
        assert rationale.counterfactual_summary.strip()
        assert rationale.evidence_summary.strip()
        assert rationale.uncertainty_statement.strip()

    def test_echo_provider_output_is_still_grounded_or_quarantined(self, agent):
        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        # Echo returns the prompt itself (not JSON) -> deterministic fallback
        rationale = RationaleGenerator(provider=EchoRationaleProvider()).generate(trace)
        assert rationale.llm_used is False
        assert rationale.grounded is True


# ------------------------------------------------------------ determinism
class TestDeterminism:
    def test_fallback_is_deterministic(self, agent):
        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        first = generate_rationale(trace)
        second = generate_rationale(trace)
        assert first.model_dump() == second.model_dump()

    def test_context_render_is_deterministic(self, agent):
        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        assert build_context(trace).render() == build_context(trace).render()

    def test_llm_used_flag_reflects_reality(self, agent):
        _, trace = agent.run_traced("Transfer ₹75,000 to beneficiary BOB-001.")
        with_llm = RationaleGenerator(provider=FakeSLM([_valid_output()])).generate(trace)
        without = generate_rationale(trace)
        assert with_llm.llm_used is True and without.llm_used is False
        assert with_llm.provenance["explain_layers_used_llm"] is False  # Q1-Q3 stayed deterministic
