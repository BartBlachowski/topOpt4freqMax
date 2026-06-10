---
name: run_full_paper_pipeline
description: Execute the full paper pipeline across literature, writing, validation, positioning, and submission readiness
---

ROLE:
Execute the full paper pipeline autonomously.

INPUT:
- Use the currently focused document as the manuscript source
- Use the project codebase (via VSCode access)
- Use references/ as the literature source
- Use existing intermediate outputs if already available
- Do not ask the user to paste sections again

ALWAYS LOAD:
- ai/STYLE_LOCK.md
- ai/WORKFLOW.md

MISSION:
Produce a submission-ready manuscript workflow outcome by:
- grounding literature claims in references/
- grounding methodology in code
- validating claims against results
- stress-testing the gap and novelty
- assessing submission readiness
- recommending journal strategy

CORE PRINCIPLE:
The paper must be written and validated from evidence, not from persuasive wording.

DO NOT:
- invent literature claims
- invent methodological details
- skip validation stages
- globally rewrite sections when local fixes suffice
- assume results support claims without checking
- overclaim novelty or significance

---

# PIPELINE OVERVIEW

The pipeline has 5 layers:

1. LITERATURE PREPARATION
2. SECTION WRITING
3. VALIDATION
4. SUBMISSION DECISION
5. JOURNAL STRATEGY

---

# STEP 0 — PRECONDITION CHECK

Confirm availability of:
- focused manuscript
- references/
- codebase access

Classify availability:

### REQUIRED
- manuscript
- references/ for Introduction / Related Work
- code for Methodology

If manuscript missing:
- STOP

If references/ missing:
- STOP before literature-driven sections

If code missing:
- STOP before Methodology pipeline

Output:
- precondition status
- missing dependencies

---

# STEP 1 — LITERATURE PREPARATION

## 1A. LITERATURE EXTRACTION
Run:
- ai/prompts/literature_extract.md

Goal:
- extract structured evidence from references/

Output:
- per-paper extraction
- method map
- assumption map
- evidence map
- gap candidates

## 1B. LITERATURE BALANCE
Run:
- ai/prompts/literature_balance.md

Goal:
- ensure balanced and representative literature coverage
- identify missing method families or narrative bias

Output:
- family coverage
- missing perspectives
- rebalancing plan

Gate:
If literature coverage is insufficient:
- mark Introduction / Related Work as literature-limited
- continue, but report risk explicitly

---

# STEP 2 — ABSTRACT PIPELINE

## 2A. WRITE ABSTRACT
Run:
- ai/prompts/write_abstract.md

## 2B. ABSTRACT CONTROL LOOP
Run in order:
- ai/prompts/abstract/abstract_audit.prompt.md
- ai/prompts/abstract/abstract_filter.prompt.md
- ai/prompts/abstract/abstract_rewrite.prompt.md
- ai/prompts/jargon_detector.prompt.md
- ai/prompts/abstract/abstract_score.prompt.md
- ai/prompts/abstract/abstract_validate.prompt.md

Decision:
- score >= 90 -> READY
- 80–89 -> one additional safe fix loop
- < 80 -> mark as NOT READY

Output:
- final abstract
- score
- validation report
- remaining risks

---

# STEP 3 — INTRODUCTION PIPELINE

Run:
- ai/run/run_introduction_pipeline.md

Expected internal stages:
- write_introduction
- gap_stress_test
- introduction_audit
- issue_filter
- safe_patch
- reviewer_simulation

Output:
- final Introduction
- status
- top reviewer attacks
- remaining risks

---

# STEP 4 — RELATED WORK PIPELINE

Run:
- ai/run/run_related_work_pipeline.md

Output:
- final Related Work
- balance status
- synthesis quality
- remaining literature risks

---

# STEP 5 — METHODOLOGY PIPELINE

Run:
- ai/run/run_methodology_pipeline.md

Expected internal stages:
- method discovery (code-first)
- write_methodology
- methodology_audit
- contribution alignment check
- issue_filter
- safe_patch
- reproducibility check
- reviewer simulation

Output:
- final Methodology
- code alignment status
- reproducibility verdict
- remaining technical gaps

---

# STEP 6 — CONCLUSION PIPELINE
Run:
- ai/run/run_conclusion_pipeline.md

Output:
- final Conclusion
- conclusion audit status
- remaining overclaim risks

---

# STEP 7 — RESULTS VALIDATION

Run:
- ai/prompts/results_validator.md

Goal:
- verify that Results support contributions and claims

Output:
- claim–evidence map
- support status
- comparison analysis
- metric analysis
- reviewer demands
- verdict

If evidence is weak:
- mark final paper as high-risk
- do not hide the issue

---

# STEP 8 — CONTRIBUTION / POSITIONING LAYER

Run in order:

## 8A. GAP STRESS TEST
- ai/prompts/gap_stress_test.md

## 8B. NOVELTY POSITIONING
- ai/prompts/novelty_positioning.md

## 8C. CONTRIBUTION REFINER
- ai/prompts/contribution_refiner.md

Goal:
- ensure the paper’s claims are real, narrow enough, and evidence-backed

Output:
- defensible gap
- novelty type
- refined contributions
- minimal positioning patches

---

# STEP 9 — GLOBAL CONSISTENCY CHECK

Run:
- ai/prompts/cross_consistency.prompt.md

Check:
- title vs abstract
- abstract vs results
- introduction vs methods
- methods vs results
- results vs conclusion
- terminology consistency
- claimed novelty vs actual content

Output:
- cross-consistency report
- severe inconsistencies
- suggested local fixes

If critical inconsistency exists:
- mark paper as NOT READY

---

# STEP 10 — FINAL SUBMISSION GUARD

Run:
- ai/prompts/final_submission_guard.md

Goal:
- simulate editor + reviewers
- decide likely submission outcome

Output:
- global assessment
- reviewer reports
- editor decision
- minimal acceptance path
- risk profile
- final verdict

---

# STEP 11 — JOURNAL STRATEGY

Run:
- ai/prompts/journal_selector.md

Goal:
- recommend best-fit journal category
- define stretch / safe targets
- align positioning

Output:
- journal archetype fit
- submission strategy
- positioning patches
- upgrade path

---

# STEP 12 — FINAL DECISION SYNTHESIS

Aggregate all major outputs into one final paper status.

Determine:

## READY FOR SUBMISSION
Conditions:
- abstract ready
- introduction ready
- methodology reproducible
- results support contributions
- gap defensible
- no critical cross-consistency failures
- final_submission_guard != reject-level

## SUBMIT AFTER MINOR FIXES
Conditions:
- no fatal scientific issues
- only local wording / consistency / positioning issues remain

## REQUIRES MAJOR REVISION
Conditions:
- methodology gaps
- evidence gaps
- weak but repairable positioning
- critical section-level weaknesses

## HIGH RISK OF REJECTION
Conditions:
- invalid gap
- unsupported contributions
- methodology not reproducible
- results do not support claims
- severe inconsistency

---

# FINAL OUTPUT FORMAT

## --- EXECUTIVE STATUS ---
Choose one:
- READY FOR SUBMISSION
- SUBMIT AFTER MINOR FIXES
- REQUIRES MAJOR REVISION
- HIGH RISK OF REJECTION

---

## --- SECTION STATUS ---

- Abstract:
- Introduction:
- Related Work:
- Methodology:
- Results support:
- Contribution validity:
- Novelty positioning:
- Cross-consistency:

Use:
- READY
- BORDERLINE
- NOT READY

---

## --- CRITICAL FINDINGS ---

List only blocking or near-blocking issues.

1.
2.
3.
4.
5.

---

## --- MINIMAL ACCEPTANCE PATH ---

### MUST FIX
- ...

### SHOULD FIX
- ...

### OPTIONAL
- ...

### DO NOT CHANGE
- ...

---

## --- REVIEWER RISK PROFILE ---

- Gap attack risk:
- Novelty attack risk:
- Methodology attack risk:
- Results attack risk:
- Consistency attack risk:

Use:
- LOW
- MEDIUM
- HIGH

---

## --- JOURNAL STRATEGY ---

- Best fit:
- Stretch target:
- Safe target:
- Avoid:
- Key positioning advice:

---

## --- ARTIFACT SUMMARY ---

List generated outputs:

- literature_extract: YES / NO
- literature_balance: YES / NO
- abstract package: YES / NO
- introduction package: YES / NO
- related work package: YES / NO
- methodology package: YES / NO
- results validation: YES / NO
- gap / novelty / contribution package: YES / NO
- final submission guard: YES / NO
- journal selector: YES / NO

---

## --- NEXT ACTION ---

Choose one:
- submit manuscript
- apply minor survival patch
- revise weak section(s)
- collect missing evidence before continuing
- retarget journal

---

## --- FAILURE HANDLING ---

If any mandatory dependency is missing:
- stop at the earliest safe point
- report exactly what is missing
- do not fabricate downstream outputs

If earlier stages contradict later stages:
- prioritize:
  1. code-grounded findings
  2. literature-grounded findings
  3. results-grounded findings
  4. prose-level claims

---

## --- SELF-CHECK ---

Confirm:
- Literature-based sections were grounded in references/
- Methodology was grounded in code
- Contributions were tested against evidence
- Final decision reflects prior audits, not optimism
- I did not hide fatal issues behind stylistic polish