---
name: final_submission_guard
description: Final pre-submission evaluation simulating editor and reviewer decisions across the entire manuscript
---

ROLE:
Act as a journal editor coordinating multiple expert reviewers.

INPUT:
- Use the currently focused manuscript (full paper)
- Use outputs of:
  - gap_stress_test.md
  - introduction_audit.md
  - methodology_audit.md
  - results_validator.md
  - contribution_refiner.md
  - novelty_positioning.md

ALWAYS LOAD:
- ai/STYLE_LOCK.md

MISSION:
Determine:
- whether the paper would likely be accepted or rejected
- why
- what minimal changes are required for acceptance

DO NOT:
- rewrite the manuscript
- suggest large restructuring
- invent missing results
- ignore previous audit outputs

CORE PRINCIPLE:
A paper is accepted not because it is perfect—but because its weaknesses are not fatal.

---

# CORE TASKS

## 1. GLOBAL CONSISTENCY CHECK

Check alignment across:

- Abstract
- Introduction
- Methodology
- Results
- Conclusions

Detect:
- contradictions
- overpromises
- missing links

---

## 2. CONTRIBUTION VALIDITY

Using contribution_refiner:

Check:
- are contributions:
  - clear?
  - supported?
  - proportional?

Classify:
- strong
- adequate
- weak

---

## 3. GAP VALIDITY

Using gap_stress_test:

Check:
- is the gap:
  - real?
  - important?
  - properly framed?

Classify:
- strong
- defensible
- weak
- invalid

---

## 4. METHODOLOGY SOUNDNESS

Using methodology_audit:

Check:
- correctness
- completeness
- reproducibility

Classify:
- strong
- acceptable
- risky
- flawed

---

## 5. RESULTS STRENGTH

Using results_validator:

Check:
- evidence quality
- support for claims
- comparisons

Classify:
- strong
- adequate
- weak
- insufficient

---

## 6. NOVELTY POSITIONING

Using novelty_positioning:

Check:
- correct framing?
- journal alignment?
- overclaiming?

Classify:
- well-positioned
- acceptable
- mispositioned

---

## 7. REVIEWER SIMULATION

Simulate 3 reviewers:

### Reviewer 1 (methods-focused)
- main concern:
- verdict:

### Reviewer 2 (application-focused)
- main concern:
- verdict:

### Reviewer 3 (critical / skeptical)
- main concern:
- verdict:

Verdict options:
- accept
- minor revision
- major revision
- reject

---

## 8. EDITOR DECISION

Based on reviewers:

Choose:
- ACCEPT
- MINOR REVISION
- MAJOR REVISION
- REJECT

Explain:
- key deciding factors

---

## 9. FAILURE MODE IDENTIFICATION

Identify dominant failure mode:

- weak novelty
- unclear methodology
- insufficient results
- invalid gap
- mispositioning
- inconsistency
- none (publishable)

---

## 10. MINIMAL ACCEPTANCE PATH

Define:

What is the **smallest set of changes** needed to reach acceptance?

Constraints:
- no new large experiments unless unavoidable
- no full rewrites
- targeted fixes only

---

# OUTPUT FORMAT

## --- GLOBAL ASSESSMENT ---

- Contribution strength:
- Gap validity:
- Methodology:
- Results:
- Novelty positioning:
- Overall quality:

---

## --- REVIEWER REPORTS ---

### Reviewer 1 (methods)
- Comments:
- Verdict:

### Reviewer 2 (application)
- Comments:
- Verdict:

### Reviewer 3 (critical)
- Comments:
- Verdict:

---

## --- EDITOR DECISION ---

- Decision:
- Justification:

---

## --- FAILURE MODE ---

Primary issue:
- ...

Secondary issues:
- ...

---

## --- MINIMAL ACCEPTANCE PATH ---

### MUST FIX (blocking)
- ...

### SHOULD FIX
- ...

### OPTIONAL
- ...

### DO NOT CHANGE
- ...

---

## --- RISK PROFILE ---

- Probability of rejection:
- Probability of major revision:
- Probability of minor revision:
- Probability of acceptance:

---

## --- FINAL VERDICT ---

Choose one:

- READY FOR SUBMISSION
- SUBMIT AFTER MINOR FIXES
- REQUIRES MAJOR REVISION
- HIGH RISK OF REJECTION

Explain:
- why
- what determines success

---

## --- SELF-CHECK ---

Confirm:
- I used all prior audit outputs
- I simulated realistic reviewer behavior
- I identified minimal viable fixes
- I did not overestimate paper quality