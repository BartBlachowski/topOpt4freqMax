---
name: conclusion_audit
description: Evaluate the Conclusion section for claim alignment, scope control, limitation correctness, and final framing
---

ROLE:
Audit the Conclusion section of the manuscript.

INPUT:
- Use the currently focused manuscript
- Read the Conclusion section only (do not infer from memory)
- Use upstream validated artifacts:
  - ai/out/positioning/contribution_refiner.md
  - ai/out/results/results_validator.md
  - ai/out/positioning/novelty_positioning.md
  - ai/out/final/cross_consistency.md (if available)

ALWAYS LOAD:
- ai/STYLE_LOCK.md
- schemas/evaluation_schema.md (if structured output is required downstream)

MISSION:
Determine whether the Conclusion is:
- correct
- aligned with validated contributions
- properly scoped
- free of overclaim
- non-generic and meaningful

CORE PRINCIPLE:
The Conclusion must be a **safe closure of validated claims**, not a place to extend or upgrade them.

DO NOT:
- rewrite the Conclusion
- fix sentences
- improve wording
- invent missing content

You are only evaluating.

---

# EVALUATION DIMENSIONS

## 1. CLAIM ALIGNMENT

Check:
- Do all claims in the Conclusion match:
  - contribution_refiner.md
  - results_validator.md

Detect:
- claims stronger than validated contributions
- claims not present in Results
- new claims introduced only in Conclusion

Classify:
- PERFECT: all claims aligned
- PARTIAL: minor drift
- BROKEN: clear mismatch

---

## 2. SCOPE CONTROL

Check:
- Does the Conclusion generalize beyond evidence?

Detect:
- “general framework” claims without evidence
- “robust across scenarios” without testing
- field-wide impact claims

Classify:
- CONTROLLED
- SLIGHTLY OVERGENERALIZED
- OVERCLAIMING

---

## 3. NOVELTY CONSISTENCY

Check:
- Does the novelty language match novelty_positioning.md?

Detect:
- method-level claims for integration work
- “novel framework” when it is combination
- implicit upgrade of contribution type

Classify:
- CONSISTENT
- DRIFT
- MISREPRESENTED

---

## 4. LIMITATION HANDLING

Check:
- Are important limitations acknowledged (if needed)?

Detect:
- missing key limitations
- over-apologetic tone
- irrelevant “future work” disguised as limitation

Classify:
- APPROPRIATE
- UNDERSTATED
- OVERSTATED
- MISSING

---

## 5. FINAL FRAMING QUALITY

Check:
- Does the Conclusion provide a clear final takeaway?

Detect:
- generic summary
- abstract repetition
- no clear resolution
- vague ending

Classify:
- STRONG
- ADEQUATE
- WEAK

---

## 6. FUTURE WORK DISCIPLINE

Check:
- Is future work:
  - minimal
  - grounded
  - non-generic

Detect:
- laundry lists
- unrelated directions
- filler sentences

Classify:
- DISCIPLINED
- GENERIC
- EXCESSIVE

---

## 7. CONSISTENCY WITH ABSTRACT & INTRODUCTION

Check:
- Does the Conclusion:
  - match Abstract claims?
  - resolve Introduction promises?

Detect:
- mismatch in contribution wording
- unresolved research question
- conclusion weaker/stronger than abstract

Classify:
- CONSISTENT
- MINOR DRIFT
- INCONSISTENT

---

# OUTPUT

## --- CONCLUSION AUDIT SUMMARY ---

- Claim alignment:
- Scope control:
- Novelty consistency:
- Limitation handling:
- Final framing:
- Future work discipline:
- Cross-section consistency:

---

## --- CRITICAL ISSUES ---

List only blocking or near-blocking problems.

1.
2.
3.
4.
5.

---

## --- OVERCLAIM DETECTION ---

List exact sentences or paraphrased claims that exceed evidence.

- ...
- ...

---

## --- UNDERCLAIM DETECTION ---

If the conclusion is too weak relative to results:

- missed valid claim:
- unnecessarily cautious statement:

---

## --- LIMITATION ANALYSIS ---

- Missing important limitation:
- Irrelevant limitation:
- Proper limitation:

---

## --- FINAL TAKEAWAY QUALITY ---

Describe whether the conclusion leaves a clear, correct impression.

---

## --- VERDICT ---

Choose one:

- READY
- BORDERLINE
- NOT_READY

---

## --- MINIMAL FIX STRATEGY ---

### MUST FIX
- ...

### SHOULD FIX
- ...

### OPTIONAL
- ...

### DO NOT CHANGE
- ...

---

## --- SELF-CHECK ---

Confirm:
- No rewriting was performed
- All issues are grounded in upstream validated artifacts
- No new claims were introduced during audit
- Overclaim detection is conservative (not speculative)