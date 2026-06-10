---
name: run_introduction_pipeline
description: Execute the full Introduction pipeline autonomously using manuscript context and extracted literature
---

ROLE:
Execute the full Introduction pipeline autonomously.

INPUT:
- Use the currently focused document as the manuscript source.
- Use the structured output of literature_extract.md as the literature evidence base.
- Use the structured output of literature_balance.md as the literature control layer.
- Do not ask the user to paste the Introduction again.

ALWAYS LOAD:
- ai/STYLE_LOCK.md
- ai/WORKFLOW.md

MISSION:
Produce a reviewer-ready Introduction that is:
- argument-driven
- literature-grounded
- aligned with the manuscript
- minimally edited during repair stages

CORE PRINCIPLE:
The Introduction must read as an argument, not an inventory.

DO NOT:
- write from memory when extracted literature is available
- skip the audit stages
- rewrite globally when local fixes suffice
- turn the Introduction into a citation dump
- overclaim novelty
- expand text unless necessary

PIPELINE:

STEP 0 — PRECONDITION CHECK
Confirm that the following inputs are available:
- manuscript in the currently focused document
- literature_extract.md output
- literature_balance.md output

If literature extraction is missing:
- STOP
- instruct the system to run literature_extract.md first

If literature balance is missing:
- STOP
- instruct the system to run literature_balance.md next

STEP 1 — WRITE INTRODUCTION
Run:
- ai/prompts/write_introduction.md

Goal:
- generate a full Introduction draft from the manuscript and structured literature extraction
- use literature_balance.md to maintain balanced coverage

Output:
- Introduction draft
- argument map
- literature usage map
- risks

STEP 2 — GAP STRESS TEST
Run:
- ai/prompts/gap_stress_test.md

Goal:
- test whether the claimed gap is real, literature-grounded, and actually addressed by the manuscript

Output:
- gap verdict
- issue list
- minimal survival patch

STEP 3 — INTRODUCTION AUDIT
Run:
- ai/prompts/introduction_audit.md

Goal:
- audit the Introduction for:
  - argument structure
  - literature grounding
  - gap validity
  - contribution positioning
  - reviewer risk

Output:
- Introduction audit
- issue list
- change classification
- top reviewer attacks
- minimal fix strategy

STEP 4 — ISSUE FILTER
Run:
- ai/prompts/issue_filter.prompt.md

Goal:
- classify issues into:
  - ESSENTIAL
  - OPTIONAL
  - HARMFUL TO CHANGE

Input:
- issue list from gap_stress_test.md
- issue list from introduction_audit.md

Output:
- change filter
- safe change set

STEP 5 — SAFE PATCH
Run:
- ai/prompts/safe_patch.prompt.md

Goal:
- apply only essential, reviewer-critical fixes
- preserve structure and tone
- avoid over-editing

Input:
- current Introduction draft
- safe change set

Output:
- revised Introduction
- applied changes
- skipped changes

STEP 6 — OPTIONAL SECOND AUDIT
Re-run:
- ai/prompts/introduction_audit.md

ONLY IF:
- critical or major issues were patched
- or the patch changed argument structure
- or gap wording was materially altered

Goal:
- verify that fixes did not damage flow or positioning

STEP 7 — REVIEWER SIMULATION
Run:
- ai/prompts/reviewer_simulation.prompt.md

Goal:
- simulate likely reviewer objections
- identify remaining attack surfaces

Output:
- 5 likely objections
- strongest strengths
- attacked sentences
- claims needing strongest defense

STEP 8 — FINAL DECISION
Decide:

IF no critical issues remain
AND the gap is defensible
AND reviewer risk is acceptable:
- mark Introduction as READY

IF only major but non-fatal issues remain:
- mark as BORDERLINE
- return the current version plus remaining risks

IF the gap is weak or unsupported
OR the argument structure fails:
- mark as NOT READY
- return minimal required next actions

DECISION RULES:
- Prefer local fixes over rewrites
- Prefer accurate narrowing over broad framing
- Prefer evidence-backed statements over persuasive wording
- If uncertain, report the uncertainty rather than inventing support

FINAL OUTPUT FORMAT:

## --- FINAL INTRODUCTION ---
<final Introduction text>

## --- PIPELINE SUMMARY ---
- Draft generated: YES / NO
- Gap stress test: PASS / PARTIAL / FAIL
- Introduction audit: PASS / PARTIAL / FAIL
- Safe patch applied: YES / NO
- Reviewer simulation completed: YES / NO

## --- STATUS ---
Choose one:
- READY
- BORDERLINE
- NOT READY

## --- APPLIED CHANGES ---
- ...

## --- REMAINING RISKS ---
- ...

## --- TOP REVIEWER ATTACKS ---
1.
2.
3.

## --- NEXT ACTION ---
Choose one:
- proceed to Methodology pipeline
- strengthen literature extraction
- strengthen gap framing
- add missing evidence before further polishing

FAILURE HANDLING:
If any required upstream artifact is missing:
- stop immediately
- report exactly what is missing
- do not hallucinate intermediate outputs

SELF-CHECK:
Confirm:
- The Introduction was written from manuscript + extracted literature
- The gap was stress-tested before finalization
- The section was audited before patching
- Only safe fixes were applied
- The final version remains argument-driven and literature-grounded