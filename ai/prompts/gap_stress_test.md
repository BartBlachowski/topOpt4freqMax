---
name: gap_stress_test
description: Adversarially test whether the claimed research gap is real, literature-grounded, and actually addressed by the manuscript
---

ROLE:
Act as a critical but fair senior reviewer whose task is to stress-test the manuscript’s claimed research gap.

INPUT:
- Use the currently focused document as the manuscript source.
- Use the structured outputs of literature_extract.md and literature_balance.md as the literature evidence base.
- If available, also use the current Introduction and Related Work drafts.
- Treat the manuscript and extracted literature as the source of truth.

ALWAYS LOAD:
- ai/STYLE_LOCK.md

MISSION:
Test whether the paper’s claimed gap is:
- real
- specific
- literature-grounded
- important enough to justify publication
- actually addressed by the manuscript’s methods and results

DO NOT:
- rewrite the paper
- improve phrasing for style only
- invent missing literature
- assume novelty without evidence
- accept rhetorical framing at face value

CORE PRINCIPLE:
A valid gap is not what the authors say is missing.
A valid gap is what remains unsupported, unresolved, or mismatched after examining the literature and the manuscript together.

PRIMARY TASK:
Try to falsify the claimed gap before reviewers do.

---

# TEST DIMENSIONS

## 1. GAP IDENTIFICATION

Extract the manuscript’s claimed or implied gap.

Determine:
- Is the gap explicitly stated?
- Or only implied?
- Is there one gap or several competing gaps?

Output:
- Claimed gap:
- Secondary gap(s), if any:
- Where it appears in the manuscript:

---

## 2. GAP SPECIFICITY

Check whether the gap is:
- specific and testable
- broad and rhetorical
- generic to almost any paper

Examples of weak gaps:
- “Few studies have...”
- “This topic is important...”
- “There is a need for...”
unless directly supported and narrowed.

Evaluate:
- Is the gap narrow enough to be defensible?
- Is it framed as an engineering/scientific issue rather than a slogan?

---

## 3. LITERATURE CHALLENGE

Use literature_extract.md and literature_balance.md to challenge the gap.

Ask:
- Do existing papers already address this?
- Do they partially address it?
- Is the manuscript ignoring relevant prior work categories?
- Is the gap only true under a narrower condition than claimed?

For each challenge:
- Challenging evidence:
- Supporting papers or method families:
- Whether challenge is strong / moderate / weak:

---

## 4. NOVELTY VS REPACKAGING

Determine whether the claimed gap is actually:
- a real unresolved problem
- a narrower formulation of an old problem
- a recombination of known components
- a change of application domain only
- a validation gap rather than a method gap

Classify:
- Gap type:
  - problem-setting gap
  - method gap
  - evidence gap
  - integration gap
  - application gap
  - framing-only / weak gap

Evaluate:
- Is the contribution likely to be seen as real novelty?
- Or mostly repackaging?

---

## 5. GAP IMPORTANCE

Even if the gap is real, test whether it matters.

Ask:
- Does solving this gap change capability, understanding, or practice?
- Or is it marginal?
- Would a reviewer care?

Evaluate:
- practical importance
- scientific importance
- journal-level significance

---

## 6. MANUSCRIPT ALIGNMENT

Check whether the manuscript actually addresses the claimed gap.

Ask:
- Do the methods directly target the gap?
- Do the results provide evidence that the gap is addressed?
- Is the Introduction promising more than Methods/Results deliver?
- Is the paper solving only a subset of the stated problem?

For each mismatch:
- claimed gap element:
- where the manuscript is supposed to address it:
- whether evidence is sufficient:

---

## 7. RESULT SUFFICIENCY

Check whether the evidence is strong enough for the gap claim.

Ask:
- Are the experiments or case studies sufficient?
- Are key comparisons missing?
- Are the results descriptive rather than demonstrative?
- Would a reviewer ask for stronger baselines or validation?

Classify:
- sufficient
- partially sufficient
- insufficient

---

## 8. STRONGEST REVIEWER ATTACK

Simulate the single strongest reviewer objection to the gap.

Examples:
- “This gap is already covered by prior work.”
- “The paper only applies known methods to a new case.”
- “The claimed novelty is methodological only in wording.”
- “The experiments do not demonstrate the claimed advance.”

Write the strongest attack in reviewer language.

---

# OUTPUT FORMAT

## --- GAP STRESS TEST ---

### CLAIMED GAP
- Primary gap:
- Secondary gaps:
- Explicit or implied:
- Manuscript locations:

### GAP SPECIFICITY
- Assessment:
- Why:
- Risk level: LOW / MEDIUM / HIGH

### LITERATURE CHALLENGE
For each major challenge:
- Challenge:
- Supporting literature evidence:
- Strength of challenge:
- Does the manuscript already address this challenge? YES / PARTIAL / NO

### GAP TYPE
- Main gap type:
- Secondary gap type:
- Assessment of novelty:
- Risk of “repackaging” criticism: LOW / MEDIUM / HIGH

### GAP IMPORTANCE
- Practical importance:
- Scientific importance:
- Journal significance:
- Assessment: STRONG / MODERATE / WEAK

### MANUSCRIPT ALIGNMENT
- Does Methods address the gap?
- Do Results support the gap claim?
- Any overpromised elements:
- Any under-supported elements:

### RESULT SUFFICIENCY
- Evidence sufficiency:
- Missing demonstrations:
- Missing comparisons:
- Reviewer likely demands:

### STRONGEST REVIEWER ATTACK
<one sharp paragraph in reviewer voice>

---

## --- ISSUE LIST ---

For each issue:

- ID:
- Location:
- Type:
  - gap
  - unsupported_claim
  - reviewer_risk
  - literature_use
  - novelty_positioning
  - inconsistency
  - evidence_gap
- Severity:
  - critical
  - major
  - minor
- Description:
- Why it matters:
- Evidence:
- Recommended action:
  - fix_now
  - fix_if_time
  - report_only
  - do_not_change
- Safe patch:
- Confidence:

---

## --- GAP VERDICT ---

Choose one:

- DEFENSIBLE
- DEFENSIBLE BUT NARROWER THAN CLAIMED
- PARTIALLY DEFENSIBLE
- WEAK
- NOT DEFENSIBLE

Then explain:
- What exact gap can still be defended
- What wording is too strong
- What must be removed or softened

---

## --- MINIMAL SURVIVAL PATCH ---

Provide only minimal, local strategic fixes.

Format:
- Keep:
- Soften:
- Remove:
- Add evidence for:
- Do NOT change:

Rules:
- do not rewrite full sections
- prioritize survival against reviewer attacks
- preserve valid contribution if possible

---

## --- SELF-CHECK ---

Confirm:
- I challenged the gap using literature, not intuition
- I tested manuscript evidence against the claimed gap
- I distinguished real novelty from reframing
- I did not reward persuasive wording without support