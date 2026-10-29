# Reading the explanations — a two-minute guide

Every TRACE-XAI run produces **one validated analysis**, rendered in **four
wordings** (TECHNICAL, BUSINESS, NORMAL, ELI5). This guide explains how the
*Explanations* page (tab 2 of the GUI) is organised, how facts and
guesses are kept apart, and where to audit everything.

## The reading order

Each explanation is written in two halves:

1. **The story** — plain sentences first: what happened, why it happened,
   and which rules it ran into (the rules appear as a small table: rule,
   result, confidence, evidence). This half uses the wording of the
   audience you picked.
2. **The proof** — shown in the labelled section directly underneath
   (you can collapse it, and expand it again at any time):
   - *Claim checks*: every claim the explanation makes, with its verdict
     (SUPPORTED / CONTRADICTED / UNKNOWN), the method used to check it, and
     the evidence refs it rests on.
   - *Validation numbers*: how many claims were checked, how many the
     records could not back (unsupported-claim rate), the overall status,
     and the guardrail decision — each glossed in plain language
     ("may be shown as-is" / "shown only with uncertainty caveats" /
     "must not be shown until corrected").
   - *The records used*: every observed value behind the explanation
     (`E1`, `E2`, …) with its exact recorded value.
   - *What we are unsure about* and *what this report cannot tell you*.

Nothing in the proof half changes when you switch audience — only the
story's wording changes.

## The four modes

| Mode | Written for | What changes |
|------|-------------|--------------|
| technical | engineers / auditors | raw ids, methods, confidences, full detail lines |
| business | risk & compliance | policy wording and business-impact sentences |
| normal | general readers | everyday language, no jargon |
| eli5 | youngest reader | short words, "the robot", "the notes" |

What **never** changes: evidence, claim texts, verdicts, counts, the
unsupported-claim rate and the guardrail. The green *cross-mode check* line
under the mode picker proves it on every run: statuses, evidence refs and
counts are identical across all four modes.

## How fact and guess stay apart

The readable panels deliberately do **not** repeat a bracket-tag on every
line. Instead:

- The story states only what the records show; anything *predicted* is
  announced in its own section — the counterfactual section opens with
  "Predictions from simulations — not recorded events."
- ELI5 analogies carry "(not evidence)" inline.
- The ✅ / ❌ / ❓ marks on rules and claims are recorded verdicts, not
  opinions, and always show their exact token (SATISFIED / VIOLATED /
  SUPPORTED / UNKNOWN …).

An UNKNOWN in a simple explanation stays an UNKNOWN. Simplifying wording
never turns a guess into a fact or a fact into a guess.

Every line still *records* its epistemic class — the audit renderings
(consolidated report, exported text and JSON) print the exact
FACT / INFERENCE / UNKNOWN label on every line.

## Where the raw records live

The story is deliberately readable, so trace paths and full machine lines
are not repeated inline. They are always one click away:

- **Tab 3 · Evidence & claims** — the full evidence index (ref → exact
  recorded value) and every claim verdict with its cited evidence.
- **Tab 7 · Export** — the same content as a text file and a JSON file
  containing the report, all four presentations, and the metrics; the
  text version labels every line FACT / INFERENCE / UNKNOWN for audit.

## Frequently asked questions

**Why does ELI5 still say UNKNOWN?**
Because the records genuinely do not say. The simplified modes re-word the
gap; they never fill it.

**Is the guardrail verdict an opinion?**
No. It is computed once from the claim verdicts (PASS / FLAG / BLOCK) and
copied into all four modes unchanged.

**Can I trust the "supported" claims?**
They are supported *by the recorded evidence of this run* — which here is
100% synthetic banking data, generated locally. The verdicts state what the
records entail, not what happened in any real bank.
