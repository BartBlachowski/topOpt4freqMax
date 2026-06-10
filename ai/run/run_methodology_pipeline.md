---
name: run_methodology_pipeline
description: Execute the full Methodology pipeline with code-grounded writing, technical audit, and reproducibility validation
---

ROLE:
Execute the full Methodology pipeline autonomously.

INPUT:
- Use the currently focused document as the manuscript
- Use the full project codebase (via VSCode access)
- Use references/ only if needed for attribution

ALWAYS LOAD:
- ai/STYLE_LOCK.md
- ai/WORKFLOW.md

MISSION:
Produce a Methodology section that is:
- technically correct
- fully consistent with implementation
- reproducible
- reviewer-proof

CORE PRINCIPLE:
The Methodology must be verifiable against code and sufficient for reimplementation.

DO NOT:
- invent algorithmic steps not present in code
- rely on abstract descriptions when concrete implementation exists
- skip audit stages
- rewrite entire sections unnecessarily
- assume “standard” steps without stating them

---

# PIPELINE

## STEP 0 — PRECONDITION CHECK

Confirm:
- manuscript is available in focused document
- codebase is accessible
- main implementation entry points exist

If code is not accessible:
- STOP
- report: “Methodology pipeline requires code access”

---

## STEP 1 — METHOD DISCOVERY (CODE-FIRST)

Inspect code:

Identify:
- main solver / runner (e.g., entry scripts)
- core algorithm functions
- data flow
- parameters and defaults
- special mechanisms (aggregation, filtering, constraints)

Output (internal):
- method structure map
- code trace map

DO NOT write Methodology yet.

---

## STEP 2 — WRITE METHODOLOGY

Run:
- ai/prompts/write_methodology.md

Goal:
- produce Methodology grounded in actual implementation
- include:
  - formulation
  - algorithm flow
  - key mechanisms
  - assumptions

Output:
- Methodology draft
- structure map
- code traceability map
- mismatch flags

---

## STEP 3 — METHODOLOGY AUDIT

Run:
- ai/prompts/methodology_audit.md

Goal:
- verify:
  - logical correctness
  - completeness
  - code consistency
  - reproducibility

Output:
- audit report
- issue list
- reproducibility gaps
- reviewer risks

---

## STEP 4 — CONTRIBUTION ALIGNMENT CHECK

Run:
- ai/prompts/contribution_refiner.md (analysis mode only)

Goal:
- ensure:
  - contributions are actually implemented in Methodology
  - no “paper-only” contributions exist

Output:
- mismatches between claims and method

---

## STEP 5 — ISSUE FILTER

Run:
- ai/prompts/issue_filter.prompt.md

Input:
- issues from methodology_audit.md
- mismatches from contribution_refiner

Goal:
- classify:
  - ESSENTIAL
  - OPTIONAL
  - HARMFUL TO CHANGE

Output:
- safe change set

---

## STEP 6 — SAFE PATCH

Run:
- ai/prompts/safe_patch.prompt.md

Goal:
- apply only critical fixes:
  - missing definitions
  - missing steps
  - inconsistencies
  - reproducibility gaps

Rules:
- do not rewrite entire Methodology
- preserve structure
- maintain alignment with code

Output:
- revised Methodology
- applied changes
- skipped changes

---

## STEP 7 — REPRODUCIBILITY CHECK

Re-evaluate:

Ask:
- could an expert reimplement this method from text alone?

If NO:
- identify missing elements
- mark as NOT READY

If PARTIAL:
- list missing elements

---

## STEP 8 — REVIEWER SIMULATION

Run:
- ai/prompts/reviewer_simulation.prompt.md

Focus on:
- technical clarity
- missing steps
- suspicious assumptions
- unclear algorithm flow

Output:
- reviewer objections
- high-risk sentences

---

## STEP 9 — FINAL DECISION

Decide:

IF:
- no critical reproducibility gaps
- code and text are aligned
- logic is consistent

→ READY

IF:
- only minor gaps remain

→ BORDERLINE

IF:
- missing steps
- code mismatch
- unclear logic

→ NOT READY

---

# DECISION RULES

- Prefer adding missing steps over rewriting
- Prefer explicit definitions over implicit assumptions
- Prefer correctness over brevity
- If code contradicts text → code wins

---

# OUTPUT FORMAT

## --- FINAL METHODOLOGY ---
<final section>

---

## --- PIPELINE SUMMARY ---

- Code inspected: YES / NO
- Method discovered: YES / NO
- Methodology written: YES / NO
- Audit performed: PASS / PARTIAL / FAIL
- Safe patch applied: YES / NO
- Reproducibility check: PASS / PARTIAL / FAIL
- Reviewer simulation: YES / NO

---

## --- STATUS ---

Choose one:
- READY
- BORDERLINE
- NOT READY

---

## --- CODE ALIGNMENT STATUS ---

- Fully aligned:
- Partially aligned:
- Misaligned components:

---

## --- APPLIED CHANGES ---

- ...

---

## --- REMAINING GAPS ---

- ...

---

## --- TOP REVIEWER ATTACKS ---

1.
2.
3.

---

## --- REPRODUCIBILITY SUMMARY ---

- Can method be reimplemented: YES / PARTIAL / NO
- Missing elements (if any):

---

## --- NEXT ACTION ---

Choose one:
- proceed to Results pipeline
- fix reproducibility gaps
- align contributions with method
- investigate code–text inconsistencies

---

## --- FAILURE HANDLING ---

If:
- code is missing
- key algorithm cannot be identified

Then:
- STOP
- report missing elements explicitly
- do NOT generate Methodology

---

## --- SELF-CHECK ---

Confirm:
- Methodology reflects actual implementation
- All major algorithmic steps are described
- No invented steps were introduced
- Reproducibility was explicitly tested
- Critical issues were fixed with minimal edits