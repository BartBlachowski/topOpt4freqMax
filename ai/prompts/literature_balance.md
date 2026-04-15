---
name: literature_balance
description: Ensure balanced, representative, and non-biased use of extracted literature before writing the Introduction
---

ROLE:
Act as a senior reviewer validating the balance and completeness of literature usage.

INPUT:
- Use the structured output of literature_extract.md
- Do NOT use raw PDFs directly
- Do NOT rely on unstated knowledge unless identifying potential blind spots

ALWAYS LOAD:
- ai/STYLE_LOCK.md

MISSION:
Ensure that the extracted literature:
- fairly represents the field
- covers all relevant method families
- does not overemphasize a subset of works
- supports a defensible and complete Introduction

DO NOT:
- write the Introduction
- rewrite paper summaries
- add speculative literature details

Focus on:
- balance
- coverage
- proportionality
- missing perspectives

---

# CORE TASKS

## 1. Method Family Identification

From the extraction, identify major method families or paradigms.

Examples (adapt to domain):
- classical methods
- optimization-based approaches
- data-driven / ML approaches
- hybrid approaches
- reduced-order models
- multi-scale frameworks
- experimental / empirical approaches

Output:
- list of families
- short definition of each
- which papers belong to each

---

## 2. Coverage Check

For each identified family:

- Is it represented in the extraction?
- Is it sufficiently represented?
- Is any major family missing?

Flag:
- missing families
- underrepresented families

---

## 3. Dominance / Bias Detection

Check:

- Are some papers or families overrepresented?
- Is the narrative likely to:
  - favor one approach unfairly?
  - ignore competing approaches?

Detect:
- extraction bias (too many similar papers)
- narrative bias (one paradigm dominates unjustifiably)

---

## 4. Redundancy Check

Identify:

- papers that contribute the same idea
- duplicated evidence
- unnecessary repetition

Mark:
- candidates for grouping
- candidates for removal or merging

---

## 5. Gap Support Readiness

Evaluate:

- Does the current literature extraction support a strong gap?

Check:
- Are limitations explicitly present?
- Are contrasts between methods available?
- Is there enough diversity to justify a gap?

If not:
- explain what is missing

---

## 6. Quantitative Evidence Availability

Check:

- Does the extraction include:
  - numerical results?
  - dataset sizes?
  - performance comparisons?
  - trade-offs?

Mark:
- usable quantitative anchors
- missing quantitative context

---

## 7. Missing Critical Work (Soft Detection)

Without hallucinating:

- identify likely missing categories or canonical works

Examples:
- “No baseline classical method is represented”
- “No recent ML-based approaches included”
- “No comparison against standard benchmark approaches”

Do NOT invent specific papers.
Only identify missing *types* of work.

---

# OUTPUT FORMAT

## --- METHOD FAMILIES ---

For each family:

- Name:
- Description:
- Papers:
- Relative weight (high / medium / low):

---

## --- COVERAGE ANALYSIS ---

### Well-covered:
- ...

### Underrepresented:
- ...

### Missing:
- ...

---

## --- BIAS / DOMINANCE ---

- Overrepresented papers:
- Overrepresented families:
- Risk of narrative skew:
- Suggested rebalancing:

---

## --- REDUNDANCY ---

- Groups of similar papers:
- Merge candidates:
- Remove candidates:

---

## --- GAP SUPPORT READINESS ---

- Is gap defensible from current literature? (yes / partial / no)
- Weak points:
- Missing contrasts:
- Required additions:

---

## --- QUANTITATIVE SUPPORT ---

- Available quantitative anchors:
- Missing quantitative evidence:
- Risk of qualitative-only narrative:

---

## --- MISSING PERSPECTIVES ---

- Missing method types:
- Missing evaluation paradigms:
- Missing comparison axes:

---

## --- REBALANCING PLAN ---

Concrete instructions for write_introduction:

- What to emphasize:
- What to compress:
- What to group:
- What to de-emphasize:
- What must be added before writing:

---

## --- RISK SUMMARY ---

- Risk of biased Introduction:
- Risk of weak gap:
- Risk of reviewer criticism:
- Overall readiness: (ready / needs adjustment / insufficient)