---
name: write_introduction
description: Write an evidence-grounded Introduction from the manuscript and structured literature extraction
---

ROLE:
Write the Introduction section for the current manuscript.

INPUT:
- Use the currently focused document as the manuscript source.
- Use the structured output of literature_extract.md as the literature source.
- Treat the manuscript and extracted literature as the sources of truth.
- Do NOT rely on unstated background memory when the manuscript or extracted literature provides the needed evidence.

ALWAYS LOAD:
- ai/STYLE_LOCK.md

MISSION:
Write an Introduction that is argument-driven, literature-grounded, and aligned with the manuscript’s actual contribution.

PRIMARY GOAL:
Construct a clear logical chain:
1. Why the problem matters
2. What existing work has achieved
3. What remains insufficient or unresolved
4. What this paper does
5. Why that contribution matters

CORE PRINCIPLE:
The Introduction must read as an argument, not an inventory.

DO NOT:
- write a citation dump
- summarize papers one by one
- imitate generic journal-introduction wording
- repeat the abstract sentence by sentence
- overclaim novelty
- include low-level implementation details
- turn the section into a methods checklist

SOURCE HIERARCHY:
1. Focused manuscript
2. Structured literature extraction from literature_extract.md
3. Nothing else

EVIDENCE RULE:
Every claim about prior work must be supportable by the extracted literature.
If the literature extraction does not support a claim, do not write it.

ARGUMENT REQUIREMENTS:
The Introduction must establish:
- the real engineering/scientific problem
- why it is nontrivial
- what categories of existing approaches exist
- what their relevant limitations are
- why those limitations motivate the present work
- what this paper contributes at a high level

MANDATORY STRUCTURE:
Write the Introduction in 4 to 6 paragraphs with this logic:

P1 — Problem context and importance
- Introduce the engineering problem.
- Explain why it matters in practice or scientifically.
- Keep this concrete, not generic.

P2 — Existing approach classes
- Present the main categories of prior work using grouped synthesis.
- Group by modeling philosophy, method family, assumption, or design strategy.
- Compare approaches functionally.
- Use quantitative narration when supported by the extraction.

P3 — Limitation / gap
- Identify what remains insufficient, inconsistent, or unresolved.
- The gap must follow directly from P2.
- The gap must be literature-grounded, not rhetorical.

P4 — This work
- State what the current paper does.
- Explain the high-level approach only.
- Position it against the gap.

P5 — Contribution / implication
- State the main contribution, capability, or significance.
- This may be merged with P4 if the section reads better, but the contribution must remain explicit.

OPTIONAL P6 — Paper roadmap
- Include only if journal style expects it or if the section genuinely benefits.
- Keep it brief.
- Avoid mechanical “Section 2 presents...” wording unless needed.

LITERATURE SYNTHESIS RULES:
- Group papers by shared logic, not by author sequence.
- Prefer category-level synthesis over individual paper listing.
- Mention individual papers only when:
  - they are foundational,
  - they define a distinct approach,
  - or they are necessary to support a specific contrast.

QUANTITATIVE NARRATION RULE:
When supported by the extracted literature, prefer statements such as:
- typical scale of datasets, studies, or experiments
- typical performance ranges
- typical tradeoffs
- typical computational or modeling limitations

Do not invent numbers.
Do not force numeric statements when the extraction does not support them.

GAP RULE:
A defensible gap must be one of the following:
- an unaddressed engineering condition
- a mismatch between existing assumptions and the current problem
- insufficient treatment of a critical constraint
- lack of evidence under relevant operating conditions
- lack of integration between method classes needed for this paper’s setting

A gap is NOT:
- “no one has studied this”
- “few studies exist”
- “this area is important”
unless directly supported by extracted evidence.

MANUSCRIPT ALIGNMENT RULE:
The Introduction must align with the manuscript’s actual content.
Do not promise:
- experiments not reported
- novelty not demonstrated
- comparisons not included
- validation not performed

STYLE RULES:
- Prefer precise, readable sentences
- Preserve scientific voice
- Avoid inflated claims
- Avoid exaggerated novelty framing
- Avoid repetitive use of “this paper”
- Avoid boilerplate transitions
- Keep narrative flow smooth

ABSTRACTION RULE:
The Introduction should explain:
- what kind of approach is used
- why it is needed
- how it differs conceptually from prior work

It should NOT explain:
- implementation sequence
- parameter settings
- solver details
- low-level algorithm mechanics
unless absolutely necessary for the gap logic

CONTRIBUTION RULE:
State the contribution in a way that is:
- specific
- defensible
- proportional to the evidence

Good contribution framing:
- what problem setting is addressed
- what methodological combination or formulation is proposed
- what practical or scientific capability is demonstrated

Avoid:
- “for the first time”
- “novel and robust”
- “state-of-the-art”
unless explicitly and safely supported

OUTPUT:

--- INTRODUCTION ---
<final introduction text>

--- ARGUMENT MAP ---
- P1 problem context:
- P2 prior work classes:
- P3 literature-grounded gap:
- P4 this work:
- P5 contribution / implication:
- Optional P6 roadmap:

--- LITERATURE USAGE MAP ---
- Group 1 used:
- Group 2 used:
- Group 3 used:
- Key contrasts used:
- Quantitative statements used:
- Papers singled out individually and why:

--- RISKS ---
- possible overclaim:
- weakly supported comparison:
- places needing citation check:
- places that may attract reviewer attack:

--- SELF-CHECK ---
Confirm:
- The section reads as an argument, not an inventory
- The gap is supported by extracted literature
- No promised contribution exceeds manuscript evidence
- No major prior-work claim depends on unstated memory