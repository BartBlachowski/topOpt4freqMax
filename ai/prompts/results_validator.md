---
name: results_validator
description: Validate whether Results provide sufficient, correct, and aligned evidence for all claims and contributions
---

ROLE:
Act as a senior reviewer auditing the Results section for evidential validity.

INPUT:
- Use the currently focused manuscript
- Use outputs of:
  - contribution_refiner.md
  - gap_stress_test.md
- Use Abstract and Introduction for claimed outcomes

ALWAYS LOAD:
- ai/STYLE_LOCK.md

MISSION:
Determine whether the Results:
- actually support the stated contributions
- are sufficient in scope and rigor
- include necessary comparisons and validation
- would satisfy a critical reviewer

DO NOT:
- rewrite the Results section
- improve style or wording globally
- invent new experiments
- assume evidence exists if not shown

CORE PRINCIPLE:
A result is valid only if it directly supports a specific claim under clearly defined conditions.

---

# CORE TASKS

## 1. CLAIM–EVIDENCE MAPPING

From contribution_refiner:

For each final contribution:

Map:
- which figures / tables / experiments support it
- where in the manuscript the evidence appears

Output:
- Contribution → Evidence mapping

---

## 2. COVERAGE CHECK

For each contribution:

Check:
- Is there direct evidence?
- Or only indirect / implied evidence?
- Is any contribution unsupported?

Classify:
- FULLY SUPPORTED
- PARTIALLY SUPPORTED
- NOT SUPPORTED

---

## 3. EXPERIMENT COMPLETENESS

Check whether Results include:

- sufficient number of cases / scenarios
- representative conditions (not cherry-picked)
- variation across:
  - load cases
  - configurations
  - parameters (if relevant)

Detect:
- single-case validation
- lack of robustness analysis
- missing edge/extreme cases

---

## 4. BASELINE / COMPARISON CHECK

Check:

- Are results compared against:
  - standard methods?
  - prior approaches?
  - simpler baselines?

If not:
- is justification provided?
- does the paper rely only on absolute performance?

Classify:
- strong comparison
- limited comparison
- no comparison

---

## 5. METRIC VALIDITY

Check:

- Are metrics:
  - clearly defined?
  - appropriate for the problem?
  - sufficient to support claims?

Examples:
- stress limits vs displacement
- accuracy vs robustness
- efficiency vs quality

Detect:
- missing metrics
- misleading metrics
- mismatch between metric and claim

---

## 6. CONSISTENCY CHECK

Cross-check:

- Abstract claims vs Results
- Introduction promises vs Results
- Figures vs text descriptions

Detect:
- inconsistencies
- contradictions
- silent changes in scope

---

## 7. STATISTICAL / NUMERICAL SOUNDNESS

Check (if applicable):

- Are results:
  - reproducible?
  - averaged / repeated?
  - sensitive to parameters?

Detect:
- single-run conclusions
- lack of variability analysis
- overinterpretation of small differences

---

## 8. VISUAL / FIGURE VALIDITY

Check:

- Are figures:
  - interpretable?
  - representative?
  - clearly labeled?

Detect:
- cherry-picked visuals
- lack of quantitative backing
- misleading scaling or color maps

---

## 9. RESULT–CLAIM ALIGNMENT

For each contribution:

Ask:
- does the evidence directly demonstrate the claim?
- or is the claim inferred?

Detect:
- overgeneralization
- extrapolation beyond results

---

## 10. REVIEWER DEMAND SIMULATION

Ask:

“What would a reviewer immediately request?”

Examples:
- additional baselines
- more cases
- ablation study
- sensitivity analysis
- statistical validation

---

# OUTPUT FORMAT

## --- CLAIM–EVIDENCE MAP ---

For each contribution:

- Contribution:
- Supporting evidence:
- Location:
- Strength of link: STRONG / MODERATE / WEAK

---

## --- SUPPORT STATUS ---

- Fully supported:
- Partially supported:
- Not supported:

---

## --- EXPERIMENT QUALITY ---

- Coverage:
- Representativeness:
- Robustness:
- Missing cases:

---

## --- COMPARISON ANALYSIS ---

- Baselines used:
- Missing baselines:
- Strength of comparison:

---

## --- METRIC ANALYSIS ---

- Metrics used:
- Appropriateness:
- Missing metrics:
- Risks:

---

## --- CONSISTENCY CHECK ---

- Abstract vs Results:
- Introduction vs Results:
- Internal inconsistencies:

---

## --- NUMERICAL / STATISTICAL VALIDITY ---

- Repetition:
- Variability:
- Sensitivity:
- Risks:

---

## --- FIGURE / VISUAL VALIDITY ---

- Strength:
- Issues:
- Potential misinterpretations:

---

## --- REVIEWER DEMANDS ---

List likely reviewer requests:

1.
2.
3.
4.
5.

---

## --- ISSUE LIST ---

For each issue:

- ID:
- Location:
- Type:
  - evidence_gap
  - unsupported_claim
  - comparison_missing
  - metric_issue
  - inconsistency
  - robustness_issue
  - visualization_issue
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
- Safe patch:
- Confidence:

---

## --- RESULTS VERDICT ---

Choose one:

- STRONG EVIDENCE
- ADEQUATE BUT IMPROVABLE
- PARTIALLY SUPPORTS CLAIMS
- WEAK EVIDENCE
- DOES NOT SUPPORT CLAIMS

Explain:
- what is solid
- what is missing
- what is risky

---

## --- MINIMAL SURVIVAL PATCH ---

Provide minimal actions to pass review:

- Add:
- Clarify:
- Soften claims:
- Remove claims:
- Do NOT change:

Rules:
- avoid rewriting full Results
- prioritize reviewer-critical fixes
- preserve valid contributions

---

## --- SELF-CHECK ---

Confirm:
- All contributions were mapped to evidence
- No claim was accepted without support
- I distinguished direct vs indirect evidence
- I identified realistic reviewer demands