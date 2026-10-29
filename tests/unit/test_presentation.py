"""Presentation tests (Innovation A): four audience modes, one validated report.

Guarantees asserted here:
1. All four modes are generated from the *identical* validated report.
2. Statuses, counts, guardrail, unsupported-claim rate and evidence refs are
   invariant across modes (wording adapts, conclusions never).
3. UNKNOWN never becomes SUPPORTED (or any positive wording) via
   simplification — checked for requirements and claims.
4. Evidence references are reachable in every mode.
5. Validated claim text is carried verbatim into every mode.
6. The presentation layer is deterministic and calls no model (no SLM/LLM
   available is the default case, not an edge case).
7. Observed facts vs inferences are visibly distinguished in every mode.
"""

import socket

import pytest

from trace_xai import TraceXAIPipeline
from trace_xai.agent.rule_agent import RuleBasedAgent
from trace_xai.banking.environment import BankEnvironment
from trace_xai.core.recorder import ExecutionRecorder
from trace_xai.presentation import (
    MODE_AUDIENCE,
    MODE_DESCRIPTION,
    ClaimKind,
    PresentedExplanation,
    PresentationMode,
    present_all,
    present_report,
    status_words,
)

MODES = list(PresentationMode)


@pytest.fixture
def pipeline() -> TraceXAIPipeline:
    return TraceXAIPipeline(agent=RuleBasedAgent(BankEnvironment.from_dataset()))


@pytest.fixture
def violation_report(pipeline):
    _, report = pipeline.run("Transfer ₹75,000 to beneficiary BOB-001.")
    return report


@pytest.fixture
def success_report(pipeline):
    _, report = pipeline.run("Transfer ₹10,000 to beneficiary BOB-001.")
    return report


def _stateless_trace():
    """Missing state_before -> requirements UNKNOWN (scenario F pattern)."""
    recorder = ExecutionRecorder(
        task_id="TASK-SYN-P", task_text="Transfer ₹10,000 to beneficiary BOB-001.",
        agent_id="qwen3.5:4b", trace_id="TRC-SYN-P",
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


def _conflicting_trace():
    """Conflicting tool outputs -> UNKNOWN, surfaced not resolved (scenario G)."""
    env = BankEnvironment.from_dataset()
    recorder = ExecutionRecorder(
        task_id="TASK-SYN-P2", task_text="Transfer ₹5,000 to beneficiary BOB-001.",
        agent_id="rule-based-agent", trace_id="TRC-SYN-P2",
    )
    recorder.record_task_received()
    recorder.record_tool_invocation(
        action="transfer", tool_name="transfer",
        tool_input={"amount_inr": 5_000, "beneficiary_id": "BOB-001"},
        state_before=env.summary(),
    )
    first = recorder.record_tool_result(
        action="transfer", tool_name="transfer",
        tool_output={"success": True, "reason_code": "OK",
                     "account_id": "ACC-SYN-001"},
        success=True,
    )
    recorder.record_tool_result(
        action="transfer", tool_name="transfer",
        tool_output={"success": False, "reason_code": "INSUFFICIENT_FUNDS",
                     "account_id": "ACC-SYN-001"},
        success=False, conflict_with_event_id=first.event_id,
    )
    recorder.record_outcome("CONFLICT", success=None)
    return recorder.finish()


# --------------------------------------------------- identical-trace invariance
class TestModesOnIdenticalTrace:
    def test_all_four_modes_generated(self, violation_report):
        modes = present_all(violation_report)
        assert set(modes) == set(MODES)
        assert all(isinstance(e, PresentedExplanation) for e in modes.values())

    def test_report_entry_points(self, violation_report):
        direct = present_report(violation_report, PresentationMode.ELI5)
        via_report = violation_report.present("eli5")
        assert via_report.model_dump() == direct.model_dump()
        all_via_report = violation_report.presented_all()
        assert set(all_via_report) == set(MODES)

    def test_verification_statuses_invariant(self, violation_report):
        digests = {m: present_report(violation_report, m).status_digest()
                   for m in MODES}
        assert len({tuple(sorted(d.items())) for d in digests.values()}) == 1

    def test_requirement_statuses_invariant(self, violation_report):
        digests = {m: present_report(violation_report, m).requirement_digest()
                   for m in MODES}
        assert len({tuple(sorted(d.items())) for d in digests.values()}) == 1

    def test_counts_rate_guardrail_invariant(self, violation_report):
        keys = {m: (present_report(violation_report, m).counts,
                    present_report(violation_report, m).unsupported_claim_rate,
                    present_report(violation_report, m).overall_status,
                    present_report(violation_report, m).guardrail)
                for m in MODES}
        assert len(set(map(str, keys.values()))) == 1

    def test_evidence_refs_invariant_and_reachable(self, violation_report):
        digests = {m: present_report(violation_report, m).ref_digest()
                   for m in MODES}
        assert len(set(digests.values())) == 1
        assert digests[MODES[0]], "some evidence refs must be reachable"
        # every mode's rendered text exposes the evidence index entries
        for mode in MODES:
            text = present_report(violation_report, mode).to_text()
            for line in violation_report.evidence[:3]:
                assert line in text, f"evidence missing in {mode} mode"

    def test_claim_text_carried_verbatim(self, violation_report):
        claims = {v.claim for v in violation_report.validation.verifications}
        assert claims
        for mode in MODES:
            presented = present_report(violation_report, mode)
            assert {c.claim for c in presented.claims} == claims
            for claim in presented.claims:
                assert claim.claim in claim.rendered

    def test_recorded_source_facts_preserved(self, violation_report):
        for mode in MODES:
            recorded = present_report(violation_report, mode).recorded
            assert recorded["decision_rationale"] \
                == violation_report.decision_rationale
            assert recorded["deviation_explanation"] \
                == violation_report.deviation.combined_explanation

    def test_wording_actually_differs_between_modes(self, violation_report):
        texts = {m: present_report(violation_report, m).to_text() for m in MODES}
        # same content conclusions, different presentation
        assert len({t for t in texts.values()}) == 4
        for mode in MODES:
            assert MODE_AUDIENCE[mode] in texts[mode]
            assert MODE_DESCRIPTION[mode]

    def test_status_wording_differs_but_status_value_does_not(self,
                                                             violation_report):
        for mode in MODES:
            row = next(r for r in present_report(violation_report, mode).requirements
                       if r.requirement_id == "REQ-AMT-001")
            assert row.status == "VIOLATED"          # invariant recorded value
            assert row.status_text == status_words(mode, "VIOLATED")
        eli5 = status_words(PresentationMode.ELI5, "VIOLATED")
        tech = status_words(PresentationMode.TECHNICAL, "VIOLATED")
        assert eli5 != tech                          # audience wording differs


# ----------------------------------------------- UNKNOWN never becomes positive
class TestUnknownNeverSimplifiedAway:
    def test_unknown_requirements_stay_unknown_in_every_mode(self, pipeline):
        report = pipeline.analyze_execution(trace=_stateless_trace())
        assert any(s == "UNKNOWN" for s in report.requirement_satisfaction.values())
        for mode in MODES:
            digest = present_report(report, mode).requirement_digest()
            assert any(s == "UNKNOWN" for s in digest.values()), mode

    def test_unknown_claims_stay_unknown_in_every_mode(self, pipeline):
        # adversarial rationale: invented refs are quarantined upstream and the
        # affected claims surface as UNKNOWN — and stay UNKNOWN in all four modes
        import json

        class FakeProvider:
            name = "fake-adversarial"
            def generate(self, system_prompt, user_prompt):
                return json.dumps({
                    "decision_rationale": "The transfer succeeded per evidence E99.",
                    "deviation_explanation": "No problems; REQ-FAKE-001 satisfied.",
                    "counterfactual_summary": "Nothing to consider.",
                    "evidence_summary": "All checks passed per E77.",
                    "uncertainty_statement": "None at all.",
                    "claims": [{"text": "Everything was fine and verified.",
                                "label": "FACT", "evidence_refs": ["E99"]}],
                })

        pipeline = TraceXAIPipeline(
            agent=RuleBasedAgent(BankEnvironment.from_dataset()),
            rationale_provider=FakeProvider(),
        )
        _, report = pipeline.run("Transfer ₹10,000 to beneficiary BOB-001.")
        assert report.validation.unsupported_claim_rate > 0
        # the invented claim itself is quarantined before validation
        assert any("E99" in n for n in report.rationale.unsupported_claims)
        invented_text = "Everything was fine and verified."
        for mode in MODES:
            presented = present_report(report, mode)
            assert all(invented_text != c.claim for c in presented.claims)
            # every verdict on an unverifiable claim stays UNKNOWN everywhere
            for claim in presented.claims:
                if "E99" in claim.claim or "REQ-FAKE-001" in claim.claim:
                    assert claim.status == "UNKNOWN", mode
                    assert claim.kind is ClaimKind.UNKNOWN

    def test_unknown_wording_is_never_positive(self):
        for mode in MODES:
            word = status_words(mode, "UNKNOWN").lower()
            for banned in ("supported", "satisfied", "passed", "compliant",
                           "backed", "verified", "okay", "met"):
                assert banned not in word, f"{mode}: {word!r}"

    def test_unknown_requirement_rows_are_tagged_unknown(self, pipeline):
        report = pipeline.analyze_execution(trace=_stateless_trace())
        for mode in MODES:
            for row in present_report(report, mode).requirements:
                if row.status == "UNKNOWN":
                    assert row.kind is ClaimKind.UNKNOWN
                else:
                    assert row.kind is ClaimKind.FACT


# ------------------------------------------- markdown rendering (UI panel)
class TestMarkdownRendering:
    """``to_markdown()`` re-renders the same recorded items as ``to_text()``.

    It must never recompute a status, drop an UNKNOWN, or lose an evidence
    reference — only the formatting differs (headings/legend vs separators).
    """

    def test_markdown_carries_same_statuses_as_text(self, violation_report):
        for mode in MODES:
            p = present_report(violation_report, mode)
            md = p.to_markdown()
            for claim, status in p.status_digest().items():
                assert status in md, f"{mode}: claim status missing in md"
                assert claim in md, f"{mode}: claim text missing in md"
            # requirements: id + the audience wording of the recorded status
            for row in p.requirements:
                assert row.requirement_id in md
                assert row.status_text in md
            # every evidence ref id AND its recorded value must be readable
            for ref in p.evidence_index[:3]:
                ref_id, sep, rest = ref.partition(" = ")
                assert sep, "evidence index lines are 'REF = value...'"
                assert f"**{ref_id}**" in md, f"{mode}: ref id missing in md"
                assert rest in md, f"{mode}: ref value missing in md"
            # the counts appear as readable numbers with their labels —
            # never as a Python dict repr
            assert str(p.counts) not in md
            count_line = next(line for line in md.splitlines()
                              if "Claims checked" in line)
            assert str(p.counts["total"]) in count_line
            for bucket in ("supported", "contradicted", "unknown"):
                assert bucket in count_line
                assert str(p.counts[bucket]) in count_line
            assert f"{p.unsupported_claim_rate:.3f}" in md
            # the guardrail decision is always glossed in plain language
            gloss = {"PASS": "may be shown as-is",
                     "FLAG": "shown only with uncertainty caveats",
                     "BLOCK": "must not be shown until corrected"}
            assert gloss[p.guardrail] in md

    def test_markdown_tells_one_story_without_duplicates(self,
                                                         violation_report):
        """Claims/evidence must appear once — not as a story block and
        again as a dedicated proof section (the old double render)."""
        for mode in MODES:
            p = present_report(violation_report, mode)
            md = p.to_markdown()
            for claim in p.status_digest():
                assert md.count(f"\"{claim}\"") == 1, \
                    f"{mode}: claim rendered twice"
            for ref in p.evidence_index:
                ref_id, _, _ = ref.partition(" = ")
                assert md.count(f"**{ref_id}**") == 1, \
                    f"{mode}: evidence ref rendered twice"

    def test_markdown_headings_follow_the_audience(self, violation_report):
        """Sections are titled with the audience wording, not jargon."""
        from trace_xai.presentation.modes import INTRO
        for mode in MODES:
            md = present_report(violation_report, mode).to_markdown()
            assert f"### {INTRO[mode]['requirements']}" in md, mode
            assert f"### {INTRO[mode]['claims']}" in md, mode
            assert f"### {INTRO[mode]['evidence']}" in md, mode
            # story-first order: the requirements section comes before the
            # proof divider that precedes the claim checks
            story = md.index(INTRO[mode]["requirements"])
            proof = md.index(INTRO[mode]["claims"])
            assert story < proof, mode

    def test_markdown_has_no_text_separator_artifacts(self, violation_report):
        for mode in MODES:
            md = present_report(violation_report, mode).to_markdown()
            # the '-- title --' plain-text separators must not leak into md
            for line in md.splitlines():
                assert not line.startswith("-- "), line
            assert "### " in md           # real headings
            assert "How to read the tags" in md

    def test_markdown_keeps_unknown_as_unknown(self, pipeline):
        """UNKNOWN must survive into the markdown with positive-free wording.

        Note the two layers: a *requirement* with status UNKNOWN is rendered
        with the audience wording for UNKNOWN; and a *claim* asserting that
        same fact ("REQ-X: UNKNOWN — no observable evidence") may itself be
        SUPPORTED, because the records do confirm the absence of evidence.
        What must never happen is an UNKNOWN being worded as satisfied.
        """
        report = pipeline.analyze_execution(trace=_stateless_trace())
        banned = ("SATISFIED", "the rule was followed", "backed by the "
                  "records", "the records back it up")
        for mode in MODES:
            p = present_report(report, mode)
            md = p.to_markdown()
            unknown_reqs = [r for r in p.requirements if r.status == "UNKNOWN"]
            assert unknown_reqs, mode
            for r in unknown_reqs:
                assert r.requirement_id in md
                # the audience wording for UNKNOWN, never a positive word
                assert r.status_text in md
                for word in banned:
                    assert word not in r.status_text, (mode, r.status_text)
            # a claim whose verdict IS unknown must never render positively
            for c in p.claims:
                if c.status == "UNKNOWN":
                    for word in ("back it up", "backed", "satisfied",
                                 "compliant"):
                        assert word not in c.rendered, (mode, c.rendered)

    def test_story_and_proof_halves_compose_the_document(self,
                                                         violation_report):
        """The UI shows story and proof in two panels: the halves must
        compose back into the exact document, losing nothing."""
        for mode in MODES:
            p = present_report(violation_report, mode)
            story, proof = p.story_markdown(), p.proof_markdown()
            assert p.to_markdown() == story + "\n\n" + proof
            # story half: legend and rules, no validation numbers
            assert "How to read the tags" in story
            assert "Claims checked" not in story
            # proof half: verdicts, numbers, every evidence ref id
            assert "Claims checked" in proof
            for claim in p.status_digest():
                assert f"\"{claim}\"" in proof
            for ref in p.evidence_index:
                ref_id, _, _ = ref.partition(" = ")
                assert f"**{ref_id}**" in proof, (mode, ref_id)
            # both halves are still real markdown with headings
            assert "### " in story and "### " in proof


# -------------------------------------------------- facts vs inference tagging
class TestFactsVersusInferences:
    def test_action_lines_are_facts(self, violation_report):
        for mode in MODES:
            block = next(b for b in present_report(violation_report, mode).blocks
                         if b.key == "action")
            assert block.lines, mode
            observed = [l for l in block.lines
                        if l.kind is ClaimKind.FACT]
            assert observed

    def test_counterfactuals_are_always_inferences(self, violation_report):
        for mode in MODES:
            block = next(b for b in present_report(violation_report, mode).blocks
                         if b.key == "counterfactuals")
            assert all(l.kind is ClaimKind.INFERENCE for l in block.lines), mode

    def test_eli5_analogy_is_labelled_and_tagged(self, violation_report):
        block = next(b for b in
                     present_report(violation_report, PresentationMode.ELI5).blocks
                     if b.key == "action")
        analogy = next(l for l in block.lines if "Analogy" in l.text)
        assert "(not evidence)" in analogy.text
        assert analogy.kind is ClaimKind.INFERENCE

    def test_business_risk_line_is_inference_not_fact(self, violation_report):
        block = next(b for b in
                     present_report(violation_report, PresentationMode.BUSINESS).blocks
                     if b.key == "deviations")
        risk = next(l for l in block.lines if "Operational risk" in l.text)
        assert risk.kind is ClaimKind.INFERENCE

    def test_every_rendered_line_carries_a_kind_tag(self, violation_report):
        for mode in MODES:
            text = present_report(violation_report, mode).to_text()
            assert "[FACT]" in text
            assert "[INFERENCE]" in text or "[UNKNOWN]" in text


# --------------------------------------------------- conflicting evidence case
class TestConflictingEvidencePresentation:
    def test_conflict_surfaced_not_resolved_in_any_mode(self, pipeline):
        report = pipeline.analyze_execution(trace=_conflicting_trace())
        conflicting = [a for a in report.requirements if a.conflicting_evidence]
        assert conflicting
        for mode in MODES:
            presented = present_report(report, mode)
            for req in presented.requirements:
                if req.status == "UNKNOWN":
                    assert req.conflicting, f"{mode}: conflict must stay visible"
                    assert req.kind is ClaimKind.UNKNOWN


# ------------------------------------------------------ deterministic fallback
class TestDeterministicFallback:
    def test_presentation_never_calls_a_model(self, violation_report):
        """Presenting with the network fully blocked must still succeed.

        The wording layer is a pure function of the validated report — no SLM
        or LLM is consulted, so presentation works in offline/no-model setups.
        """
        real_socket = socket.socket

        class NoNetwork(real_socket):
            def connect(self, *a, **k):  # pragma: no cover - guard
                raise AssertionError("presentation layer must not open sockets")

        socket.socket = NoNetwork
        try:
            for mode in MODES:
                text = present_report(violation_report, mode).to_text()
                assert text
        finally:
            socket.socket = real_socket

    def test_presentation_is_deterministic(self, violation_report):
        for mode in MODES:
            a = present_report(violation_report, mode).model_dump()
            b = present_report(violation_report, mode).model_dump()
            assert a == b

    def test_provenance_records_no_model_use(self, violation_report):
        for mode in MODES:
            provenance = present_report(violation_report, mode).provenance
            assert provenance["presentation_llm_used"] is False
            assert provenance["presentation_generator"]
            assert provenance["synthetic_data"] is True

    def test_invalid_mode_rejected(self, violation_report):
        with pytest.raises(ValueError):
            present_report(violation_report, "shakespeare")
