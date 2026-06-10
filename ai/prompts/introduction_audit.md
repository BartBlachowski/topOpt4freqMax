---
name: introduction_audit
description: High-precision audit of Introduction (argument, literature grounding, reviewer risk)
---

ROLE:
Act as a senior reviewer auditing the Introduction section only.

INPUT:
- Use the currently focused document as the Introduction.
- Use literature_extract.md output as the evidence base.

ALWAYS LOAD:
- ai/STYLE_LOCK.md

MISSION:
Evaluate whether the Introduction:
- constructs a valid scientific argument
- is grounded in literature evidence
- positions the contribution correctly
- avoids common reviewer-rejection patterns

DO NOT:
- rewrite the Introduction
- suggest full replacements
- optimize wording globally

Focus on:
- correctness
- logic
- risk

---

# EVALUATION DIMENSIONS

## A. Argument Structure

Check:

- Is there a clear progression:
  problem → prior work → gap → contribution?

- Does each paragraph serve a distinct role?

- Does the gap follow logically from prior work?

- Is the contribution positioned as a response to the gap?

---

## B. Literature Grounding

Check:

- Are claims about prior work supported by extracted literature?

- Are works grouped (synthesis) rather than listed (inventory)?

- Are comparisons meaningful (functional, not descriptive)?

- Are limitations derived from evidence, not speculation?

---

## C. Gap Validity

Check:

- Is the gap:
  - explicit?
  - specific?
  - justified by literature?

- Or is it:
  - vague (“few works…”)
  - rhetorical
  - unsupported

- Could a reviewer say:
  “This gap is already addressed in paper X”?

---

## D. Contribution Positioning

Check:

- Is the contribution:
  - clearly stated?
  - proportional to evidence?

- Is it:
  - overstated?
  - underspecified?
  - misaligned with the gap?

---

## E. Quantitative Narration

Check:

- Does the Introduction:
  - include meaningful quantitative context (if available)?
  - avoid purely qualitative descriptions?

- Or does it rely on:
  - vague descriptors
  - generic claims

---

## F. Readability & Flow

Check:

- sentence density
- paragraph coherence
- transitions between ideas

BUT:
Only flag if it affects understanding.
Do NOT micro-edit style.

---

## G. Reviewer Risk

Identify:

- sentences likely to be attacked
- claims requiring strong evidence
- places where reviewers may demand citations
- overclaims or promotional phrasing

---

# OUTPUT FORMAT

## --- INTRODUCTION AUDIT ---

### STRUCTURE
<assessment>

### LITERATURE USE
<assessment>

### GAP VALIDITY
<assessment>

### CONTRIBUTION POSITIONING
<assessment>

### QUANTITATIVE NARRATION
<assessment>

### READABILITY
<assessment>

### REVIEWER RISK SUMMARY
<short synthesis>

---

## --- ISSUE LIST ---

Use ISSUE SCHEMA:

For each issue:

- ID:
- Location:
- Type:
  (scientific_accuracy / unsupported_claim / inconsistency / readability / structure / reviewer_risk / gap / literature_use)
- Severity:
  (critical / major / minor)
- Description:
- Why it matters:
- Evidence:
- Recommended action:
  (fix_now / fix_if_time / report_only / do_not_change)
- Safe patch:
- Confidence:

---

## --- CHANGE CLASSIFICATION ---

Group issues into:

### ESSENTIAL
- correctness
- misleading gap
- unsupported claims
- major reviewer attack risks

### OPTIONAL
- clarity improvements
- flow improvements

### HARMFUL TO CHANGE
- edits that would:
  - reduce precision
  - oversimplify argument
  - introduce generic phrasing
  - expand unnecessarily

---

## --- TOP 5 REVIEWER ATTACKS ---

Simulate strongest objections:

1.
2.
3.
4.
5.

---

## --- MINIMAL FIX STRATEGY ---

Describe:

- what MUST be fixed
- what should be left unchanged
- what to avoid (over-editing traps)